// Function: sub_2F1D0D0
// Address: 0x2f1d0d0
//
__int64 __fastcall sub_2F1D0D0(__int64 a1, unsigned __int64 *a2)
{
  int v3; // ebx
  __int64 v4; // r12
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // r12
  __int64 v8; // r13
  _QWORD *v9; // r9
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // rax
  __int64 v12; // rsi
  _QWORD *v14; // rax
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // [rsp+8h] [rbp-68h]
  _QWORD *v21; // [rsp+10h] [rbp-60h]
  _QWORD *v22; // [rsp+10h] [rbp-60h]
  _QWORD *v23; // [rsp+10h] [rbp-60h]
  _QWORD *v24; // [rsp+10h] [rbp-60h]
  _QWORD *v25; // [rsp+10h] [rbp-60h]
  _QWORD *v26; // [rsp+18h] [rbp-58h]
  _QWORD *v27; // [rsp+18h] [rbp-58h]
  _QWORD *v28; // [rsp+18h] [rbp-58h]
  _QWORD *v29; // [rsp+18h] [rbp-58h]
  _QWORD *v30; // [rsp+18h] [rbp-58h]
  __int64 v32; // [rsp+28h] [rbp-48h]
  __int64 v33[7]; // [rsp+38h] [rbp-38h] BYREF

  v3 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 24LL))(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    v3 = -858993459 * ((__int64)(a2[1] - *a2) >> 6);
  if ( v3 )
  {
    v4 = (unsigned int)(v3 - 1);
    v5 = 0;
    v6 = v4 + 2;
    v7 = 1;
    v32 = v6;
    do
    {
      while ( 1 )
      {
        v8 = v5 + 320;
        if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD, __int64 *))(*(_QWORD *)a1 + 32LL))(
               a1,
               (unsigned int)(v7 - 1),
               v33) )
        {
          break;
        }
        v5 += 320;
        if ( ++v7 == v32 )
          return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 48LL))(a1);
      }
      v9 = (_QWORD *)a2[1];
      v10 = *a2;
      v11 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)((__int64)v9 - *a2) >> 6);
      if ( v11 <= v7 - 1 )
      {
        if ( v11 < v7 )
        {
          sub_2F1CAB0(a2, v7 - v11);
          v10 = *a2;
        }
        else if ( v11 > v7 )
        {
          v14 = (_QWORD *)(v10 + v8);
          v20 = v10 + v8;
          if ( v9 != (_QWORD *)(v10 + v8) )
          {
            do
            {
              v15 = v14[34];
              if ( (_QWORD *)v15 != v14 + 36 )
              {
                v21 = v9;
                v26 = v14;
                j_j___libc_free_0(v15);
                v9 = v21;
                v14 = v26;
              }
              v16 = v14[28];
              if ( (_QWORD *)v16 != v14 + 30 )
              {
                v22 = v9;
                v27 = v14;
                j_j___libc_free_0(v16);
                v9 = v22;
                v14 = v27;
              }
              v17 = v14[22];
              if ( (_QWORD *)v17 != v14 + 24 )
              {
                v23 = v9;
                v28 = v14;
                j_j___libc_free_0(v17);
                v9 = v23;
                v14 = v28;
              }
              v18 = v14[13];
              if ( (_QWORD *)v18 != v14 + 15 )
              {
                v24 = v9;
                v29 = v14;
                j_j___libc_free_0(v18);
                v9 = v24;
                v14 = v29;
              }
              v19 = v14[3];
              if ( (_QWORD *)v19 != v14 + 5 )
              {
                v25 = v9;
                v30 = v14;
                j_j___libc_free_0(v19);
                v9 = v25;
                v14 = v30;
              }
              v14 += 40;
            }
            while ( v9 != v14 );
            v10 = *a2;
            a2[1] = v20;
          }
        }
      }
      ++v7;
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 144LL))(a1);
      v12 = v10 + v5;
      v5 += 320;
      sub_2F0EDE0(a1, v12);
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 152LL))(a1);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 40LL))(a1, v33[0]);
    }
    while ( v7 != v32 );
  }
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 48LL))(a1);
}
