// Function: sub_18843B0
// Address: 0x18843b0
//
__int64 __fastcall sub_18843B0(__int64 a1, __int64 *a2)
{
  __int64 v2; // r15
  int v3; // ebx
  __int64 v4; // rax
  __int64 v5; // r13
  unsigned __int64 v6; // rbx
  __int64 v7; // r14
  char i; // al
  __int64 v9; // rsi
  __int64 v10; // rdi
  unsigned __int64 v11; // rax
  _QWORD *v13; // r15
  _QWORD *v14; // r12
  __int64 v15; // r13
  __int64 v16; // r14
  __int64 v17; // rdi
  __int64 v18; // r13
  __int64 v19; // r14
  __int64 v20; // rdi
  __int64 v21; // rdi
  __int64 v22; // rdi
  __int64 v23; // rdi
  __int64 v24; // rdi
  __int64 v25; // [rsp+8h] [rbp-78h]
  __int64 v26; // [rsp+10h] [rbp-70h]
  __int64 v27; // [rsp+18h] [rbp-68h]
  __int64 v29; // [rsp+30h] [rbp-50h]
  __int64 v30; // [rsp+38h] [rbp-48h]
  _QWORD v31[7]; // [rsp+48h] [rbp-38h] BYREF

  v2 = a1;
  v3 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 24LL))(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    v3 = 678152731 * ((a2[1] - *a2) >> 3);
  if ( v3 )
  {
    v4 = (unsigned int)(v3 - 1);
    v5 = 0;
    v6 = 1;
    v7 = a1;
    v29 = v4 + 2;
    for ( i = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD *))(*(_QWORD *)v7 + 32LL))(v7, 0, v31);
          ;
          i = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD *))(*(_QWORD *)v7 + 32LL))(
                v7,
                (unsigned int)(v6 - 1),
                v31) )
    {
      v30 = v5 + 152;
      if ( i )
      {
        v9 = *a2;
        v10 = a2[1];
        v11 = 0x86BCA1AF286BCA1BLL * ((v10 - *a2) >> 3);
        if ( v11 <= v6 - 1 )
        {
          if ( v11 < v6 )
          {
            sub_1882EC0(a2, v6 - v11);
            v9 = *a2;
          }
          else if ( v11 > v6 )
          {
            v13 = (_QWORD *)a2[1];
            v27 = v9 + v5 + 152;
            if ( v10 != v27 )
            {
              v26 = v5;
              v14 = (_QWORD *)(v9 + v5 + 152);
              v25 = v7;
              do
              {
                v15 = v14[17];
                v16 = v14[16];
                if ( v15 != v16 )
                {
                  do
                  {
                    v17 = *(_QWORD *)(v16 + 16);
                    if ( v17 )
                      j_j___libc_free_0(v17, *(_QWORD *)(v16 + 32) - v17);
                    v16 += 40;
                  }
                  while ( v15 != v16 );
                  v16 = v14[16];
                }
                if ( v16 )
                  j_j___libc_free_0(v16, v14[18] - v16);
                v18 = v14[14];
                v19 = v14[13];
                if ( v18 != v19 )
                {
                  do
                  {
                    v20 = *(_QWORD *)(v19 + 16);
                    if ( v20 )
                      j_j___libc_free_0(v20, *(_QWORD *)(v19 + 32) - v20);
                    v19 += 40;
                  }
                  while ( v18 != v19 );
                  v19 = v14[13];
                }
                if ( v19 )
                  j_j___libc_free_0(v19, v14[15] - v19);
                v21 = v14[10];
                if ( v21 )
                  j_j___libc_free_0(v21, v14[12] - v21);
                v22 = v14[7];
                if ( v22 )
                  j_j___libc_free_0(v22, v14[9] - v22);
                v23 = v14[4];
                if ( v23 )
                  j_j___libc_free_0(v23, v14[6] - v23);
                v24 = v14[1];
                if ( v24 )
                  j_j___libc_free_0(v24, v14[3] - v24);
                v14 += 19;
              }
              while ( v13 != v14 );
              v5 = v26;
              v7 = v25;
              a2[1] = v27;
              v9 = *a2;
            }
          }
        }
        ++v6;
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v7 + 104LL))(v7);
        sub_1883FF0(v7, v9 + v5);
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v7 + 112LL))(v7);
        (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v7 + 40LL))(v7, v31[0]);
        v5 = v30;
        if ( v29 == v6 )
        {
LABEL_10:
          v2 = v7;
          return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v2 + 48LL))(v2);
        }
      }
      else
      {
        v5 += 152;
        if ( v29 == ++v6 )
          goto LABEL_10;
      }
    }
  }
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v2 + 48LL))(v2);
}
