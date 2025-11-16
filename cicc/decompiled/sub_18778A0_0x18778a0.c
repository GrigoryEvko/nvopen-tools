// Function: sub_18778A0
// Address: 0x18778a0
//
void __fastcall sub_18778A0(__int64 a1, _QWORD *a2)
{
  _QWORD *v2; // rbx
  unsigned __int64 v3; // rax
  __int64 v4; // rcx
  _QWORD *v5; // r12
  _QWORD *v6; // r13
  int v7; // eax
  __int64 v8; // rax
  _QWORD *v9; // r14
  __int64 v10; // rdx
  _QWORD *v11; // r13
  unsigned __int64 v12; // r12
  __int64 v13; // r15
  __int64 v14; // rdi
  __int64 v15; // rdx
  int v16; // ecx
  __int64 v17; // rdx
  _QWORD *v18; // rdx
  __int64 v19; // r12
  __int64 v20; // rdi
  __int64 v21; // rdx
  int *v22; // rcx
  int v23; // eax
  __int64 v24; // rax
  __int64 v25; // rcx
  __int64 v26; // r15
  __int64 v27; // rdi
  bool v28; // zf
  _QWORD *v29; // r14
  __int64 v30; // rcx
  __int64 v31; // rax
  __int64 v32; // rcx
  __int64 v33; // rax
  __int64 v34; // r15
  __int64 v35; // rdi
  __int64 v36; // rax
  int v37; // edx
  int v39; // [rsp+28h] [rbp-78h] BYREF
  __int64 v40; // [rsp+30h] [rbp-70h]
  int *v41; // [rsp+38h] [rbp-68h]
  int *v42; // [rsp+40h] [rbp-60h]
  __int64 v43; // [rsp+48h] [rbp-58h]
  unsigned __int64 v44; // [rsp+50h] [rbp-50h]
  __int64 v45; // [rsp+58h] [rbp-48h]
  __int64 v46; // [rsp+60h] [rbp-40h]
  __int64 v47; // [rsp+68h] [rbp-38h]

  if ( (_QWORD *)a1 != a2 && a2 != (_QWORD *)(a1 + 80) )
  {
    v2 = (_QWORD *)(a1 + 88);
    do
    {
      v3 = v2[5];
      v4 = v2[1];
      v5 = v2 - 1;
      v6 = v2;
      if ( v3 <= *(_QWORD *)(a1 + 48) )
      {
        if ( v4 )
        {
          v23 = *(_DWORD *)v2;
          v40 = v2[1];
          v39 = v23;
          v41 = (int *)v2[2];
          v42 = (int *)v2[3];
          *(_QWORD *)(v4 + 8) = &v39;
          v24 = v2[4];
          v2[1] = 0;
          v2[2] = v2;
          v2[3] = v2;
          v2[4] = 0;
          v43 = v24;
          v3 = v2[5];
        }
        else
        {
          v39 = 0;
          v40 = 0;
          v41 = &v39;
          v42 = &v39;
          v43 = 0;
        }
        v25 = v2[6];
        v44 = v3;
        v45 = v25;
        v46 = v2[7];
        v47 = v2[8];
        if ( v3 <= *(v2 - 5) )
        {
          v29 = v2;
        }
        else
        {
          v26 = 0;
          while ( 1 )
          {
            v5 = v6 - 11;
            while ( v26 )
            {
              sub_1876060(*(_QWORD *)(v26 + 24));
              v27 = v26;
              v26 = *(_QWORD *)(v26 + 16);
              j_j___libc_free_0(v27, 40);
            }
            v28 = *(v6 - 9) == 0;
            v6[1] = 0;
            v29 = v6 - 10;
            v6[2] = v6;
            v6[3] = v6;
            v6[4] = 0;
            if ( !v28 )
            {
              v30 = v29[2];
              *(_DWORD *)v6 = *((_DWORD *)v6 - 20);
              v31 = v29[1];
              v29[12] = v30;
              v32 = v29[3];
              v29[11] = v31;
              v29[13] = v32;
              *(_QWORD *)(v31 + 8) = v6;
              v33 = v29[4];
              v29[1] = 0;
              v29[14] = v33;
              v29[2] = v29;
              v29[3] = v29;
              v29[4] = 0;
            }
            v29[15] = v29[5];
            v29[16] = v29[6];
            v29[17] = v29[7];
            v29[18] = v29[8];
            if ( v44 <= *(v6 - 15) )
              break;
            v26 = v29[1];
            v6 -= 10;
          }
          v34 = v5[2];
          while ( v34 )
          {
            sub_1876060(*(_QWORD *)(v34 + 24));
            v35 = v34;
            v34 = *(_QWORD *)(v34 + 16);
            j_j___libc_free_0(v35, 40);
          }
        }
        v36 = v40;
        v5[3] = v29;
        v5[2] = 0;
        v5[4] = v29;
        v5[5] = 0;
        if ( v36 )
        {
          v37 = v39;
          v5[2] = v36;
          *((_DWORD *)v5 + 2) = v37;
          v5[3] = v41;
          v5[4] = v42;
          *(_QWORD *)(v36 + 8) = v29;
          v5[5] = v43;
        }
        v9 = v2 + 9;
        v5[6] = v44;
        v5[7] = v45;
        v5[8] = v46;
        v5[9] = v47;
      }
      else
      {
        if ( v4 )
        {
          v7 = *(_DWORD *)v2;
          v40 = v2[1];
          v39 = v7;
          v41 = (int *)v2[2];
          v42 = (int *)v2[3];
          *(_QWORD *)(v4 + 8) = &v39;
          v8 = v2[4];
          v2[1] = 0;
          v2[2] = v2;
          v2[3] = v2;
          v2[4] = 0;
          v43 = v8;
          v3 = v2[5];
        }
        else
        {
          v39 = 0;
          v40 = 0;
          v41 = &v39;
          v42 = &v39;
          v43 = 0;
        }
        v44 = v3;
        v9 = v2 + 9;
        v10 = (__int64)v5 - a1;
        v11 = v2;
        v45 = v2[6];
        v46 = v2[7];
        v47 = v2[8];
        v12 = 0xCCCCCCCCCCCCCCCDLL * (((__int64)v5 - a1) >> 4);
        if ( v10 > 0 )
        {
          do
          {
            v13 = v11[1];
            while ( v13 )
            {
              sub_1876060(*(_QWORD *)(v13 + 24));
              v14 = v13;
              v13 = *(_QWORD *)(v13 + 16);
              j_j___libc_free_0(v14, 40);
            }
            v15 = *(v11 - 9);
            v11[1] = 0;
            v11[2] = v11;
            v11[3] = v11;
            v11[4] = 0;
            if ( v15 )
            {
              v16 = *((_DWORD *)v11 - 20);
              v11[1] = v15;
              *(_DWORD *)v11 = v16;
              v11[2] = *(v11 - 8);
              v11[3] = *(v11 - 7);
              *(_QWORD *)(v15 + 8) = v11;
              v17 = *(v11 - 6);
              *(v11 - 9) = 0;
              v11[4] = v17;
              v18 = v11 - 10;
              *(v11 - 8) = v11 - 10;
              *(v11 - 7) = v11 - 10;
              *(v11 - 6) = 0;
            }
            else
            {
              v18 = v11 - 10;
            }
            v11[5] = *(v11 - 5);
            v11[6] = *(v11 - 4);
            v11[7] = *(v11 - 3);
            v11[8] = *(v11 - 2);
            v11 = v18;
            --v12;
          }
          while ( v12 );
        }
        v19 = *(_QWORD *)(a1 + 16);
        while ( v19 )
        {
          sub_1876060(*(_QWORD *)(v19 + 24));
          v20 = v19;
          v19 = *(_QWORD *)(v19 + 16);
          j_j___libc_free_0(v20, 40);
        }
        v21 = v40;
        *(_QWORD *)(a1 + 16) = 0;
        *(_QWORD *)(a1 + 24) = a1 + 8;
        *(_QWORD *)(a1 + 32) = a1 + 8;
        *(_QWORD *)(a1 + 40) = 0;
        if ( v21 )
        {
          *(_DWORD *)(a1 + 8) = v39;
          v22 = v41;
          *(_QWORD *)(a1 + 16) = v21;
          *(_QWORD *)(a1 + 24) = v22;
          *(_QWORD *)(a1 + 32) = v42;
          *(_QWORD *)(v21 + 8) = a1 + 8;
          *(_QWORD *)(a1 + 40) = v43;
        }
        *(_QWORD *)(a1 + 48) = v44;
        *(_QWORD *)(a1 + 56) = v45;
        *(_QWORD *)(a1 + 64) = v46;
        *(_QWORD *)(a1 + 72) = v47;
      }
      v2 += 10;
    }
    while ( a2 != v9 );
  }
}
