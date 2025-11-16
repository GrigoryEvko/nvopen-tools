// Function: sub_342E160
// Address: 0x342e160
//
void __fastcall sub_342E160(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, _BYTE *a6)
{
  __int64 v8; // r14
  __int64 v9; // rdi
  unsigned __int8 *v10; // r12
  unsigned __int8 **v11; // rax
  unsigned __int8 **v12; // rdx
  __int64 v13; // r12
  int v14; // edx
  __int64 v15; // rbx
  int v16; // r13d
  __int64 *v17; // rax

  *(_BYTE *)(*(_QWORD *)(a1 + 64) + 762LL) = 0;
  if ( a2 != a4 )
  {
    v8 = a2;
    do
    {
      v9 = *(_QWORD *)(a1 + 72);
      if ( *(_BYTE *)(v9 + 1016) )
        goto LABEL_14;
      v10 = (unsigned __int8 *)(v8 - 24);
      if ( !v8 )
        v10 = 0;
      if ( *(_BYTE *)(a1 + 852) )
      {
        v11 = *(unsigned __int8 ***)(a1 + 832);
        v12 = &v11[*(unsigned int *)(a1 + 844)];
        if ( v11 == v12 )
          goto LABEL_18;
        while ( v10 != *v11 )
        {
          if ( v12 == ++v11 )
            goto LABEL_18;
        }
      }
      else
      {
        v17 = sub_C8CA60(a1 + 824, (__int64)v10);
        v9 = *(_QWORD *)(a1 + 72);
        if ( !v17 )
        {
LABEL_18:
          a2 = (__int64)v10;
          sub_33C77F0(v9, v10);
          goto LABEL_12;
        }
      }
      a2 = (__int64)v10;
      sub_3387170(v9, (__int64)v10);
LABEL_12:
      v8 = *(_QWORD *)(v8 + 8);
    }
    while ( a4 != v8 );
  }
  v9 = *(_QWORD *)(a1 + 72);
LABEL_14:
  v13 = *(_QWORD *)(a1 + 64);
  v15 = sub_3373A60(v9, a2, a3, a4, a5, (__int64)a6);
  v16 = v14;
  if ( v15 )
  {
    nullsub_1875();
    *(_QWORD *)(v13 + 384) = v15;
    *(_DWORD *)(v13 + 392) = v16;
    sub_33E2B60();
  }
  else
  {
    *(_QWORD *)(v13 + 384) = 0;
    *(_DWORD *)(v13 + 392) = v14;
  }
  *a6 = *(_BYTE *)(*(_QWORD *)(a1 + 72) + 1016LL);
  sub_3381FB0(*(_QWORD *)(a1 + 72));
  sub_3373070(*(_QWORD *)(a1 + 72));
  sub_342DBC0(a1);
}
