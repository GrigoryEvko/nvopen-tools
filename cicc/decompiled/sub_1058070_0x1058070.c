// Function: sub_1058070
// Address: 0x1058070
//
void __fastcall sub_1058070(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  unsigned __int8 *v7; // r12
  __int64 *v8; // rax
  __int64 *v9; // rdx

  v6 = *(_QWORD *)(a2 + 16);
  if ( v6 )
  {
    while ( 1 )
    {
      v7 = *(unsigned __int8 **)(v6 + 24);
      if ( *v7 <= 0x1Cu )
        goto LABEL_8;
      if ( *(_BYTE *)(a1 + 1284) )
      {
        v8 = *(__int64 **)(a1 + 1264);
        v9 = &v8[*(unsigned int *)(a1 + 1276)];
        if ( v8 == v9 )
          goto LABEL_11;
        while ( v7 != (unsigned __int8 *)*v8 )
        {
          if ( v9 == ++v8 )
            goto LABEL_11;
        }
LABEL_8:
        v6 = *(_QWORD *)(v6 + 8);
        if ( !v6 )
          return;
      }
      else
      {
        if ( sub_C8CA60(a1 + 1256, *(_QWORD *)(v6 + 24)) )
          goto LABEL_8;
LABEL_11:
        sub_1057F60(a1, v7, v9, a4, a5, a6);
        v6 = *(_QWORD *)(v6 + 8);
        if ( !v6 )
          return;
      }
    }
  }
}
