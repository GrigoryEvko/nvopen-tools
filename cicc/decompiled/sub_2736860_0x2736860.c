// Function: sub_2736860
// Address: 0x2736860
//
__int64 __fastcall sub_2736860(__int64 *a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rdx
  __int64 v4; // rbx
  __int64 v5; // rcx
  unsigned int v6; // eax
  __int64 v7; // r15
  unsigned __int8 *v8; // rbx
  __int64 v9; // rdi
  __int64 v10; // rsi
  __int64 i; // [rsp+8h] [rbp-58h]
  __int64 v13; // [rsp+10h] [rbp-50h] BYREF
  __int64 v14; // [rsp+18h] [rbp-48h]
  __int64 v15; // [rsp+20h] [rbp-40h]
  unsigned int v16; // [rsp+28h] [rbp-38h]

  v2 = *(_QWORD *)(a2 + 80);
  v13 = 0;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  if ( v2 == a2 + 72 )
  {
    v9 = 0;
    v10 = 0;
  }
  else
  {
    do
    {
      v3 = a1[1];
      if ( v2 )
      {
        v4 = v2 - 24;
        v5 = (unsigned int)(*(_DWORD *)(v2 + 20) + 1);
        v6 = *(_DWORD *)(v2 + 20) + 1;
      }
      else
      {
        v4 = 0;
        v5 = 0;
        v6 = 0;
      }
      if ( v6 < *(_DWORD *)(v3 + 32) )
      {
        if ( *(_QWORD *)(*(_QWORD *)(v3 + 24) + 8 * v5) )
        {
          v7 = *(_QWORD *)(v4 + 56);
          for ( i = v4 + 48; i != v7; v7 = *(_QWORD *)(v7 + 8) )
          {
            while ( 1 )
            {
              v8 = 0;
              if ( v7 )
                v8 = (unsigned __int8 *)(v7 - 24);
              if ( !(unsigned __int8)sub_DFB0F0(*a1) )
                break;
              v7 = *(_QWORD *)(v7 + 8);
              if ( i == v7 )
                goto LABEL_13;
            }
            sub_27367D0((__int64)a1, (__int64)&v13, v8);
          }
        }
      }
LABEL_13:
      v2 = *(_QWORD *)(v2 + 8);
    }
    while ( a2 + 72 != v2 );
    v9 = v14;
    v10 = 16LL * v16;
  }
  return sub_C7D6A0(v9, v10, 8);
}
