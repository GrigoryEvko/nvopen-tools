// Function: sub_25FA3C0
// Address: 0x25fa3c0
//
void __fastcall sub_25FA3C0(unsigned __int64 *a1, unsigned __int64 *a2)
{
  unsigned __int64 v2; // rdx
  unsigned __int64 v3; // rax
  unsigned __int64 *v4; // rbx
  _QWORD *v5; // rdi
  unsigned __int64 v6; // r9
  unsigned __int64 v7; // r14
  unsigned __int64 v8; // r12
  unsigned __int64 v9; // r13
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // r15
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rdi
  unsigned __int64 v16; // r12
  unsigned __int64 v17; // rbx
  unsigned __int64 v18; // r14
  __int64 v19; // rsi
  __int64 v20; // rdi
  unsigned __int64 v21; // rdx
  unsigned __int64 *v22; // rbx
  unsigned __int64 v23; // r12
  unsigned __int64 v24; // r13
  unsigned __int64 v25; // rax
  unsigned __int64 *v26; // r14
  unsigned __int64 v27; // r15
  unsigned __int64 v28; // rax
  __int64 v29; // rsi
  __int64 v30; // rdi
  unsigned __int64 v31; // rbx
  __int64 v32; // rsi
  __int64 v33; // rdi
  unsigned __int64 v36; // [rsp+10h] [rbp-60h]
  unsigned __int64 v37; // [rsp+18h] [rbp-58h]
  unsigned __int64 *v38; // [rsp+20h] [rbp-50h]
  unsigned __int64 v39; // [rsp+28h] [rbp-48h]
  unsigned __int64 v40; // [rsp+30h] [rbp-40h]

  if ( a1 != a2 )
  {
    v38 = a1 + 3;
    while ( a2 != v38 )
    {
      v2 = *v38;
      v3 = v38[1];
      v4 = v38;
      v5 = v38 + 3;
      v6 = v38[2];
      v38 += 3;
      v37 = v3;
      v40 = v2;
      v36 = v6;
      v39 = 0x86BCA1AF286BCA1BLL * ((__int64)(v3 - v2) >> 3);
      if ( v39 * *(unsigned int *)(v2 + 4) <= *(unsigned int *)(*a1 + 4)
                                            * 0x86BCA1AF286BCA1BLL
                                            * ((__int64)(a1[1] - *a1) >> 3) )
      {
        *(v38 - 1) = 0;
        v21 = *(v38 - 6);
        *(v38 - 2) = 0;
        *(v38 - 3) = 0;
        if ( v39 * *(unsigned int *)(v40 + 4) <= *(unsigned int *)(v21 + 4)
                                               * 0x86BCA1AF286BCA1BLL
                                               * ((__int64)(*(v38 - 5) - v21) >> 3) )
        {
          *v4 = v40;
          v4[1] = v3;
          v4[2] = v6;
        }
        else
        {
          v22 = v38 - 6;
          v23 = 0;
          v24 = 0;
          do
          {
            v25 = v22[1];
            v22[3] = v21;
            v26 = v22;
            v27 = v24;
            *v22 = 0;
            v22[4] = v25;
            v28 = v22[2];
            v22[1] = 0;
            v22[5] = v28;
            v22[2] = 0;
            while ( v27 != v23 )
            {
              v29 = *(unsigned int *)(v27 + 144);
              v30 = *(_QWORD *)(v27 + 128);
              v27 += 152LL;
              sub_C7D6A0(v30, 8 * v29, 4);
              sub_C7D6A0(*(_QWORD *)(v27 - 56), 8LL * *(unsigned int *)(v27 - 40), 4);
              sub_C7D6A0(*(_QWORD *)(v27 - 88), 16LL * *(unsigned int *)(v27 - 72), 8);
              sub_C7D6A0(*(_QWORD *)(v27 - 120), 16LL * *(unsigned int *)(v27 - 104), 8);
            }
            if ( v24 )
              j_j___libc_free_0(v24);
            v21 = *(v22 - 3);
            v24 = *v22;
            v22 -= 3;
            v23 = v22[4];
          }
          while ( v39 * *(unsigned int *)(v40 + 4) > *(unsigned int *)(v21 + 4)
                                                   * 0x86BCA1AF286BCA1BLL
                                                   * ((__int64)(*(v26 - 2) - v21) >> 3) );
          *v26 = v40;
          v26[1] = v37;
          v26[2] = v36;
          if ( v24 != v23 )
          {
            v31 = v24;
            do
            {
              v32 = *(unsigned int *)(v31 + 144);
              v33 = *(_QWORD *)(v31 + 128);
              v31 += 152LL;
              sub_C7D6A0(v33, 8 * v32, 4);
              sub_C7D6A0(*(_QWORD *)(v31 - 56), 8LL * *(unsigned int *)(v31 - 40), 4);
              sub_C7D6A0(*(_QWORD *)(v31 - 88), 16LL * *(unsigned int *)(v31 - 72), 8);
              sub_C7D6A0(*(_QWORD *)(v31 - 120), 16LL * *(unsigned int *)(v31 - 104), 8);
            }
            while ( v31 != v23 );
          }
          if ( v24 )
            j_j___libc_free_0(v24);
        }
      }
      else
      {
        *(v5 - 1) = 0;
        *(v5 - 2) = 0;
        *(v5 - 3) = 0;
        v7 = 0xAAAAAAAAAAAAAAABLL * (v4 - a1);
        if ( (char *)v4 - (char *)a1 > 0 )
        {
          v8 = 0;
          v9 = 0;
          while ( 1 )
          {
            v10 = *(v4 - 3);
            v4 -= 3;
            v11 = v9;
            *v4 = 0;
            v4[3] = v10;
            v12 = v4[1];
            v4[1] = 0;
            v4[4] = v12;
            v13 = v4[2];
            v4[2] = 0;
            v4[5] = v13;
            while ( v11 != v8 )
            {
              v14 = *(unsigned int *)(v11 + 144);
              v15 = *(_QWORD *)(v11 + 128);
              v11 += 152LL;
              sub_C7D6A0(v15, 8 * v14, 4);
              sub_C7D6A0(*(_QWORD *)(v11 - 56), 8LL * *(unsigned int *)(v11 - 40), 4);
              sub_C7D6A0(*(_QWORD *)(v11 - 88), 16LL * *(unsigned int *)(v11 - 72), 8);
              sub_C7D6A0(*(_QWORD *)(v11 - 120), 16LL * *(unsigned int *)(v11 - 104), 8);
            }
            if ( v9 )
              j_j___libc_free_0(v9);
            if ( !--v7 )
              break;
            v9 = *v4;
            v8 = v4[1];
          }
        }
        v16 = *a1;
        v17 = a1[1];
        *a1 = v40;
        v18 = v16;
        a1[1] = v37;
        a1[2] = v36;
        while ( v17 != v18 )
        {
          v19 = *(unsigned int *)(v18 + 144);
          v20 = *(_QWORD *)(v18 + 128);
          v18 += 152LL;
          sub_C7D6A0(v20, 8 * v19, 4);
          sub_C7D6A0(*(_QWORD *)(v18 - 56), 8LL * *(unsigned int *)(v18 - 40), 4);
          sub_C7D6A0(*(_QWORD *)(v18 - 88), 16LL * *(unsigned int *)(v18 - 72), 8);
          sub_C7D6A0(*(_QWORD *)(v18 - 120), 16LL * *(unsigned int *)(v18 - 104), 8);
        }
        if ( v16 )
          j_j___libc_free_0(v16);
      }
    }
  }
}
