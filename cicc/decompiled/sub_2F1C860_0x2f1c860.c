// Function: sub_2F1C860
// Address: 0x2f1c860
//
__int64 __fastcall sub_2F1C860(__int64 a1, unsigned __int64 *a2)
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
  unsigned __int64 v19; // [rsp+8h] [rbp-68h]
  _QWORD *v20; // [rsp+10h] [rbp-60h]
  _QWORD *v21; // [rsp+10h] [rbp-60h]
  _QWORD *v22; // [rsp+10h] [rbp-60h]
  _QWORD *v23; // [rsp+10h] [rbp-60h]
  _QWORD *v24; // [rsp+18h] [rbp-58h]
  _QWORD *v25; // [rsp+18h] [rbp-58h]
  _QWORD *v26; // [rsp+18h] [rbp-58h]
  _QWORD *v27; // [rsp+18h] [rbp-58h]
  __int64 v29; // [rsp+28h] [rbp-48h]
  __int64 v30[7]; // [rsp+38h] [rbp-38h] BYREF

  v3 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 24LL))(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    v3 = 1041204193 * ((__int64)(a2[1] - *a2) >> 3);
  if ( v3 )
  {
    v4 = (unsigned int)(v3 - 1);
    v5 = 0;
    v6 = v4 + 2;
    v7 = 1;
    v29 = v6;
    do
    {
      while ( 1 )
      {
        v8 = v5 + 264;
        if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD, __int64 *))(*(_QWORD *)a1 + 32LL))(
               a1,
               (unsigned int)(v7 - 1),
               v30) )
        {
          break;
        }
        v5 += 264;
        if ( v29 == ++v7 )
          return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 48LL))(a1);
      }
      v9 = (_QWORD *)a2[1];
      v10 = *a2;
      v11 = 0xF83E0F83E0F83E1LL * ((__int64)((__int64)v9 - *a2) >> 3);
      if ( v11 <= v7 - 1 )
      {
        if ( v11 < v7 )
        {
          sub_2F1C300(a2, v7 - v11);
          v10 = *a2;
        }
        else if ( v11 > v7 )
        {
          v14 = (_QWORD *)(v10 + v8);
          v19 = v10 + v8;
          if ( v9 != (_QWORD *)(v10 + v8) )
          {
            do
            {
              v15 = v14[27];
              if ( (_QWORD *)v15 != v14 + 29 )
              {
                v20 = v9;
                v24 = v14;
                j_j___libc_free_0(v15);
                v9 = v20;
                v14 = v24;
              }
              v16 = v14[21];
              if ( (_QWORD *)v16 != v14 + 23 )
              {
                v21 = v9;
                v25 = v14;
                j_j___libc_free_0(v16);
                v9 = v21;
                v14 = v25;
              }
              v17 = v14[15];
              if ( (_QWORD *)v17 != v14 + 17 )
              {
                v22 = v9;
                v26 = v14;
                j_j___libc_free_0(v17);
                v9 = v22;
                v14 = v26;
              }
              v18 = v14[8];
              if ( (_QWORD *)v18 != v14 + 10 )
              {
                v23 = v9;
                v27 = v14;
                j_j___libc_free_0(v18);
                v9 = v23;
                v14 = v27;
              }
              v14 += 33;
            }
            while ( v9 != v14 );
            v10 = *a2;
            a2[1] = v19;
          }
        }
      }
      ++v7;
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 144LL))(a1);
      v12 = v10 + v5;
      v5 += 264;
      sub_2F0F4A0(a1, v12);
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 152LL))(a1);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 40LL))(a1, v30[0]);
    }
    while ( v29 != v7 );
  }
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 48LL))(a1);
}
