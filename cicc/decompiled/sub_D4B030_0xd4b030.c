// Function: sub_D4B030
// Address: 0xd4b030
//
__int64 __fastcall sub_D4B030(__int64 a1)
{
  __int64 v1; // r12
  int v2; // r13d
  unsigned int v3; // r15d
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 v6; // rdx
  _QWORD *v7; // rax
  _QWORD *v8; // rcx
  __int64 *v10; // rax
  __int64 v11; // [rsp+8h] [rbp-68h]
  __int64 v12; // [rsp+18h] [rbp-58h] BYREF
  __int64 v13; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v14; // [rsp+28h] [rbp-48h]
  int v15; // [rsp+38h] [rbp-38h]

  v1 = 0;
  v12 = sub_D47930(a1);
  sub_B1C7C0((__int64)&v13, &v12);
  v2 = v15;
  v3 = v14;
  v4 = v13;
  if ( v14 != v15 )
  {
    while ( 1 )
    {
      v5 = sub_B46EC0(v4, v3);
      v6 = v5;
      if ( *(_BYTE *)(a1 + 84) )
      {
        v7 = *(_QWORD **)(a1 + 64);
        v8 = &v7[*(unsigned int *)(a1 + 76)];
        if ( v7 != v8 )
        {
          while ( v6 != *v7 )
          {
            if ( v8 == ++v7 )
              goto LABEL_10;
          }
          goto LABEL_7;
        }
LABEL_10:
        if ( !v6 )
          goto LABEL_7;
        if ( v1 )
          return 0;
        ++v3;
        v1 = v6;
        if ( v2 == v3 )
          return v1;
      }
      else
      {
        v11 = v5;
        v10 = sub_C8CA60(a1 + 56, v5);
        v6 = v11;
        if ( !v10 )
          goto LABEL_10;
LABEL_7:
        if ( v2 == ++v3 )
          return v1;
      }
    }
  }
  return 0;
}
