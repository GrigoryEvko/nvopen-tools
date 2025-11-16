// Function: sub_2F1F520
// Address: 0x2f1f520
//
__int64 __fastcall sub_2F1F520(__int64 a1, unsigned __int64 *a2)
{
  int v3; // ebx
  __int64 v4; // r14
  __int64 v5; // rax
  unsigned __int64 v6; // rbx
  char v7; // al
  __int64 v8; // rdx
  unsigned __int64 *v9; // r10
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // rax
  unsigned __int64 *v13; // rax
  unsigned __int64 v14; // [rsp+8h] [rbp-68h]
  __int64 v15; // [rsp+10h] [rbp-60h]
  unsigned __int64 *v16; // [rsp+18h] [rbp-58h]
  __int64 v17; // [rsp+20h] [rbp-50h]
  unsigned __int64 *v18; // [rsp+20h] [rbp-50h]
  __int64 v19; // [rsp+28h] [rbp-48h]
  __int64 v20[7]; // [rsp+38h] [rbp-38h] BYREF

  v3 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 24LL))(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    v3 = -1431655765 * ((__int64)(a2[1] - *a2) >> 4);
  if ( v3 )
  {
    v4 = 0;
    v5 = (unsigned int)(v3 - 1) + 2LL;
    v6 = 1;
    v19 = v5;
    do
    {
      while ( 1 )
      {
        v7 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64 *))(*(_QWORD *)a1 + 32LL))(
               a1,
               (unsigned int)(v6 - 1),
               v20);
        v8 = v4 + 48;
        if ( v7 )
          break;
        v4 += 48;
        if ( ++v6 == v19 )
          return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 48LL))(a1);
      }
      v9 = (unsigned __int64 *)a2[1];
      v10 = *a2;
      v11 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)v9 - *a2) >> 4);
      if ( v11 <= v6 - 1 )
      {
        if ( v6 > v11 )
        {
          sub_2F1F2A0(a2, v6 - v11);
          v10 = *a2;
          v8 = v4 + 48;
        }
        else if ( v6 < v11 )
        {
          v13 = (unsigned __int64 *)(v10 + v8);
          v14 = v10 + v8;
          if ( v9 != (unsigned __int64 *)(v10 + v8) )
          {
            do
            {
              if ( (unsigned __int64 *)*v13 != v13 + 2 )
              {
                v15 = v8;
                v16 = v9;
                v18 = v13;
                j_j___libc_free_0(*v13);
                v8 = v15;
                v9 = v16;
                v13 = v18;
              }
              v13 += 6;
            }
            while ( v9 != v13 );
            v10 = *a2;
            a2[1] = v14;
          }
        }
      }
      v17 = v8;
      ++v6;
      sub_2F0E9C0(a1, v4 + v10);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 40LL))(a1, v20[0]);
      v4 = v17;
    }
    while ( v6 != v19 );
  }
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 48LL))(a1);
}
