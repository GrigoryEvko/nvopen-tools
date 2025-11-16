// Function: sub_38727B0
// Address: 0x38727b0
//
__int64 __fastcall sub_38727B0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v6; // rbx
  unsigned __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 *v15; // [rsp+18h] [rbp-68h]
  __int64 *v16; // [rsp+20h] [rbp-60h] BYREF
  __int64 v17; // [rsp+28h] [rbp-58h]
  _BYTE v18[80]; // [rsp+30h] [rbp-50h] BYREF

  v16 = (__int64 *)v18;
  v17 = 0x400000000LL;
  sub_13F9CA0(a5, (__int64)&v16);
  v6 = v16;
  v15 = &v16[(unsigned int)v17];
  if ( v15 != v16 )
  {
    do
    {
      v7 = sub_157EBA0(*v6);
      if ( *(_BYTE *)(v7 + 16) == 26 && (*(_DWORD *)(v7 + 20) & 0xFFFFFFF) == 3 )
      {
        v8 = *(_QWORD *)(v7 - 72);
        if ( *(_BYTE *)(v8 + 16) == 75 )
        {
          v9 = *(_QWORD *)(v8 - 48);
          if ( *(_BYTE *)(v9 + 16) > 0x17u )
          {
            v10 = *(_QWORD *)(v8 - 24);
            if ( *(_BYTE *)(v10 + 16) > 0x17u )
            {
              if ( a3 == sub_146F1B0(*a2, *(_QWORD *)(v8 - 48)) && sub_15CCEE0(*(_QWORD *)(*a2 + 56), v9, a4) )
              {
                *(_BYTE *)(a1 + 16) = 1;
                *(_QWORD *)a1 = v9;
                *(_QWORD *)(a1 + 8) = 0;
                goto LABEL_15;
              }
              if ( a3 == sub_146F1B0(*a2, v10) && sub_15CCEE0(*(_QWORD *)(*a2 + 56), v10, a4) )
              {
                *(_BYTE *)(a1 + 16) = 1;
                *(_QWORD *)a1 = v10;
                *(_QWORD *)(a1 + 8) = 0;
                goto LABEL_15;
              }
            }
          }
        }
      }
      ++v6;
    }
    while ( v15 != v6 );
  }
  v11 = sub_3872670((__int64)a2, a3, a4);
  if ( v11 )
  {
    *(_BYTE *)(a1 + 16) = 1;
    *(_QWORD *)a1 = v11;
    *(_QWORD *)(a1 + 8) = v12;
  }
  else
  {
    *(_BYTE *)(a1 + 16) = 0;
  }
LABEL_15:
  if ( v16 != (__int64 *)v18 )
    _libc_free((unsigned __int64)v16);
  return a1;
}
