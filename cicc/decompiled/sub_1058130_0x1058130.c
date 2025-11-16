// Function: sub_1058130
// Address: 0x1058130
//
__int64 __fastcall sub_1058130(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // r13
  __int64 v5; // rbx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 *v9; // rax
  __int64 *v10; // rdx

  result = sub_AA5930(a2);
  if ( result != v3 )
  {
    v4 = v3;
    v5 = result;
    do
    {
      if ( !(unsigned __int8)sub_E45350(v5) )
      {
        if ( *(_BYTE *)(a1 + 1284) )
        {
          v9 = *(__int64 **)(a1 + 1264);
          v10 = &v9[*(unsigned int *)(a1 + 1276)];
          if ( v9 != v10 )
          {
            while ( v5 != *v9 )
            {
              if ( v10 == ++v9 )
                goto LABEL_16;
            }
            goto LABEL_9;
          }
LABEL_16:
          sub_1057F60(a1, (unsigned __int8 *)v5, v10, v6, v7, v8);
          goto LABEL_9;
        }
        if ( !sub_C8CA60(a1 + 1256, v5) )
          goto LABEL_16;
      }
LABEL_9:
      if ( !v5 )
        BUG();
      result = *(_QWORD *)(v5 + 32);
      if ( !result )
        BUG();
      v5 = 0;
      if ( *(_BYTE *)(result - 24) == 84 )
        v5 = result - 24;
    }
    while ( v4 != v5 );
  }
  return result;
}
