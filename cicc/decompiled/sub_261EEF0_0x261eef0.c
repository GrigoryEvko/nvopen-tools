// Function: sub_261EEF0
// Address: 0x261eef0
//
void __fastcall sub_261EEF0(__int64 a1, _QWORD *a2)
{
  _QWORD *v3; // rbx
  __int64 v4; // rax
  _DWORD *v5; // rsi
  _QWORD *v6; // rdx
  int v7; // edx
  unsigned __int64 v8; // rax
  __int64 v9; // rsi
  _QWORD *v10; // r15
  unsigned __int64 v11; // r12
  unsigned __int64 v12; // r14
  unsigned __int64 v13; // rdi
  __int64 v14; // rdx
  int v15; // ecx
  __int64 v16; // rdx
  _QWORD *v17; // rdx
  unsigned __int64 v18; // r12
  unsigned __int64 v19; // rdi
  __int64 v20; // rdx
  int v21; // ecx
  int v22; // ecx
  unsigned __int64 v23; // rax
  _DWORD *i; // rcx
  __int64 v25; // r8
  __int64 v26; // rdi
  __int64 v27; // r8
  __int64 v28; // rdx
  unsigned __int64 v29; // rdx
  _QWORD *v30; // rax
  bool v31; // zf
  _DWORD *v32; // r9
  __int64 v33; // rax
  int v34; // edx
  _QWORD *v36; // [rsp+18h] [rbp-68h]
  int v37; // [rsp+28h] [rbp-58h] BYREF
  __int64 v38; // [rsp+30h] [rbp-50h]
  int *v39; // [rsp+38h] [rbp-48h]
  int *v40; // [rsp+40h] [rbp-40h]
  unsigned __int64 v41; // [rsp+48h] [rbp-38h]

  if ( (_QWORD *)a1 != a2 && a2 != (_QWORD *)(a1 + 48) )
  {
    v3 = (_QWORD *)(a1 + 56);
    do
    {
      v4 = v3[1];
      v5 = v3 - 1;
      v6 = v3;
      if ( v3[4] >= *(_QWORD *)(a1 + 40) )
      {
        if ( v4 )
        {
          v22 = *(_DWORD *)v3;
          v38 = v3[1];
          v37 = v22;
          v39 = (int *)v3[2];
          v40 = (int *)v3[3];
          *(_QWORD *)(v4 + 8) = &v37;
          v23 = v3[4];
          v3[1] = 0;
          v41 = v23;
          v3[2] = v3;
          v3[3] = v3;
          v3[4] = 0;
          if ( *(v3 - 2) > v23 )
            goto LABEL_22;
        }
        else
        {
          v31 = *(v3 - 2) == 0;
          v37 = 0;
          v38 = 0;
          v39 = &v37;
          v40 = &v37;
          v41 = 0;
          if ( !v31 )
          {
LABEL_22:
            v5 = v3 - 7;
            for ( i = v3; ; i -= 12 )
            {
              v30 = i - 12;
              v31 = *(v6 - 5) == 0;
              v6[1] = 0;
              v6[2] = v6;
              v32 = i - 12;
              v6[3] = v6;
              v6[4] = 0;
              if ( v31 )
              {
                if ( v41 >= *((_QWORD *)i - 8) )
                  goto LABEL_27;
              }
              else
              {
                v25 = v30[2];
                *i = *(i - 12);
                v26 = v30[1];
                v30[8] = v25;
                v27 = v30[3];
                v30[7] = v26;
                v30[9] = v27;
                *(_QWORD *)(v26 + 8) = v6;
                v28 = v30[4];
                v30[1] = 0;
                v30[10] = v28;
                v29 = *((_QWORD *)i - 8);
                v30[2] = v30;
                v30[3] = v30;
                v30[4] = 0;
                if ( v41 >= v29 )
                  goto LABEL_27;
              }
              v5 = i - 26;
              v6 = i - 12;
            }
          }
        }
        v32 = v3;
LABEL_27:
        v33 = v38;
        *((_QWORD *)v5 + 2) = 0;
        *((_QWORD *)v5 + 3) = v32;
        *((_QWORD *)v5 + 4) = v32;
        *((_QWORD *)v5 + 5) = 0;
        if ( v33 )
        {
          v34 = v37;
          *((_QWORD *)v5 + 2) = v33;
          v5[2] = v34;
          *((_QWORD *)v5 + 3) = v39;
          *((_QWORD *)v5 + 4) = v40;
          *(_QWORD *)(v33 + 8) = v32;
          *((_QWORD *)v5 + 5) = v41;
        }
        v36 = v3 + 5;
      }
      else
      {
        if ( v4 )
        {
          v7 = *(_DWORD *)v3;
          v38 = v3[1];
          v37 = v7;
          v39 = (int *)v3[2];
          v40 = (int *)v3[3];
          *(_QWORD *)(v4 + 8) = &v37;
          v8 = v3[4];
          v3[1] = 0;
          v41 = v8;
          v3[2] = v3;
          v3[3] = v3;
          v3[4] = 0;
        }
        else
        {
          v37 = 0;
          v38 = 0;
          v39 = &v37;
          v40 = &v37;
          v41 = 0;
        }
        v9 = (__int64)v5 - a1;
        v10 = v3;
        v36 = v3 + 5;
        v11 = 0xAAAAAAAAAAAAAAABLL * (v9 >> 4);
        if ( v9 > 0 )
        {
          do
          {
            v12 = v10[1];
            while ( v12 )
            {
              sub_261DCB0(*(_QWORD *)(v12 + 24));
              v13 = v12;
              v12 = *(_QWORD *)(v12 + 16);
              j_j___libc_free_0(v13);
            }
            v14 = *(v10 - 5);
            v10[1] = 0;
            v10[2] = v10;
            v10[3] = v10;
            v10[4] = 0;
            if ( v14 )
            {
              v15 = *((_DWORD *)v10 - 12);
              v10[1] = v14;
              *(_DWORD *)v10 = v15;
              v10[2] = *(v10 - 4);
              v10[3] = *(v10 - 3);
              *(_QWORD *)(v14 + 8) = v10;
              v16 = *(v10 - 2);
              *(v10 - 5) = 0;
              v10[4] = v16;
              v17 = v10 - 6;
              *(v10 - 4) = v10 - 6;
              *(v10 - 3) = v10 - 6;
              *(v10 - 2) = 0;
            }
            else
            {
              v17 = v10 - 6;
            }
            v10 = v17;
            --v11;
          }
          while ( v11 );
        }
        v18 = *(_QWORD *)(a1 + 16);
        while ( v18 )
        {
          sub_261DCB0(*(_QWORD *)(v18 + 24));
          v19 = v18;
          v18 = *(_QWORD *)(v18 + 16);
          j_j___libc_free_0(v19);
        }
        v20 = v38;
        *(_QWORD *)(a1 + 16) = 0;
        *(_QWORD *)(a1 + 24) = a1 + 8;
        *(_QWORD *)(a1 + 32) = a1 + 8;
        *(_QWORD *)(a1 + 40) = 0;
        if ( v20 )
        {
          v21 = v37;
          *(_QWORD *)(a1 + 16) = v20;
          *(_DWORD *)(a1 + 8) = v21;
          *(_QWORD *)(a1 + 24) = v39;
          *(_QWORD *)(a1 + 32) = v40;
          *(_QWORD *)(v20 + 8) = a1 + 8;
          *(_QWORD *)(a1 + 40) = v41;
        }
      }
      v3 += 6;
    }
    while ( a2 != v36 );
  }
}
