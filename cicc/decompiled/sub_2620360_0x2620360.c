// Function: sub_2620360
// Address: 0x2620360
//
void __fastcall sub_2620360(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  __int64 v6; // rbx
  __int64 v7; // r14
  __int64 v8; // r9
  int *v9; // r10
  __int64 v10; // r11
  __int64 v11; // r15
  int *v12; // r13
  unsigned __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rdx
  int v16; // eax
  __int64 v17; // rax
  __int64 v18; // r12
  __int64 v19; // rax
  __int64 v20; // rax
  bool v21; // zf
  __int64 v22; // rax
  unsigned __int64 v23; // rbx
  unsigned __int64 v24; // rdi
  __int64 v25; // rax
  int v26; // edx
  __int64 v27; // [rsp+0h] [rbp-A0h]
  int *v28; // [rsp+8h] [rbp-98h]
  __int64 v29; // [rsp+10h] [rbp-90h]
  __int64 v30; // [rsp+18h] [rbp-88h]
  int v31; // [rsp+28h] [rbp-78h] BYREF
  __int64 v32; // [rsp+30h] [rbp-70h]
  int *v33; // [rsp+38h] [rbp-68h]
  int *v34; // [rsp+40h] [rbp-60h]
  __int64 v35; // [rsp+48h] [rbp-58h]
  unsigned __int64 v36; // [rsp+50h] [rbp-50h]
  __int64 v37; // [rsp+58h] [rbp-48h]
  __int64 v38; // [rsp+60h] [rbp-40h]
  __int64 v39; // [rsp+68h] [rbp-38h]

  while ( 1 )
  {
    v30 = a3;
    if ( !a4 )
      break;
    v5 = a5;
    if ( !a5 )
      break;
    v6 = a4;
    if ( a4 + a5 == 2 )
    {
      v13 = *(_QWORD *)(a1 + 48);
      if ( *(_QWORD *)(a2 + 48) > v13 )
      {
        v14 = *(_QWORD *)(a1 + 16);
        v15 = a1 + 8;
        if ( v14 )
        {
          v16 = *(_DWORD *)(a1 + 8);
          v32 = *(_QWORD *)(a1 + 16);
          v31 = v16;
          v33 = *(int **)(a1 + 24);
          v34 = *(int **)(a1 + 32);
          *(_QWORD *)(v14 + 8) = &v31;
          v35 = *(_QWORD *)(a1 + 40);
          v13 = *(_QWORD *)(a1 + 48);
        }
        else
        {
          v31 = 0;
          v32 = 0;
          v33 = &v31;
          v34 = &v31;
          v35 = 0;
        }
        v36 = v13;
        v17 = *(_QWORD *)(a1 + 56);
        v18 = a2 + 8;
        *(_QWORD *)(a1 + 16) = 0;
        v37 = v17;
        v19 = *(_QWORD *)(a1 + 64);
        *(_QWORD *)(a1 + 24) = v15;
        v38 = v19;
        v20 = *(_QWORD *)(a1 + 72);
        *(_QWORD *)(a1 + 32) = v15;
        *(_QWORD *)(a1 + 40) = 0;
        v21 = *(_QWORD *)(a2 + 16) == 0;
        v39 = v20;
        if ( !v21 )
        {
          *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
          v22 = *(_QWORD *)(a2 + 16);
          *(_QWORD *)(a1 + 16) = v22;
          *(_QWORD *)(a1 + 24) = *(_QWORD *)(a2 + 24);
          *(_QWORD *)(a1 + 32) = *(_QWORD *)(a2 + 32);
          *(_QWORD *)(v22 + 8) = v15;
          *(_QWORD *)(a1 + 40) = *(_QWORD *)(a2 + 40);
          *(_QWORD *)(a2 + 16) = 0;
          *(_QWORD *)(a2 + 24) = v18;
          *(_QWORD *)(a2 + 32) = v18;
          *(_QWORD *)(a2 + 40) = 0;
        }
        *(_QWORD *)(a1 + 48) = *(_QWORD *)(a2 + 48);
        *(_QWORD *)(a1 + 56) = *(_QWORD *)(a2 + 56);
        *(_QWORD *)(a1 + 64) = *(_QWORD *)(a2 + 64);
        *(_QWORD *)(a1 + 72) = *(_QWORD *)(a2 + 72);
        v23 = *(_QWORD *)(a2 + 16);
        while ( v23 )
        {
          sub_261DCB0(*(_QWORD *)(v23 + 24));
          v24 = v23;
          v23 = *(_QWORD *)(v23 + 16);
          j_j___libc_free_0(v24);
        }
        v25 = v32;
        *(_QWORD *)(a2 + 16) = 0;
        *(_QWORD *)(a2 + 24) = v18;
        *(_QWORD *)(a2 + 32) = v18;
        *(_QWORD *)(a2 + 40) = 0;
        if ( v25 )
        {
          v26 = v31;
          *(_QWORD *)(a2 + 16) = v25;
          *(_DWORD *)(a2 + 8) = v26;
          *(_QWORD *)(a2 + 24) = v33;
          *(_QWORD *)(a2 + 32) = v34;
          *(_QWORD *)(v25 + 8) = v18;
          *(_QWORD *)(a2 + 40) = v35;
        }
        *(_QWORD *)(a2 + 48) = v36;
        *(_QWORD *)(a2 + 56) = v37;
        *(_QWORD *)(a2 + 64) = v38;
        *(_QWORD *)(a2 + 72) = v39;
      }
      return;
    }
    if ( a4 > a5 )
    {
      v11 = a4 / 2;
      v9 = (int *)sub_261AB30(a2, a3, a1 + 80 * (a4 / 2));
      v7 = 0xCCCCCCCCCCCCCCCDLL * (((__int64)v9 - a2) >> 4);
    }
    else
    {
      v7 = a5 / 2;
      v10 = sub_261ABA0(a1, a2, a2 + 80 * (a5 / 2));
      v11 = 0xCCCCCCCCCCCCCCCDLL * ((v10 - a1) >> 4);
    }
    v28 = v9;
    v27 = v8;
    v29 = v10;
    v12 = sub_261FC50(v10, (int *)a2, v9);
    sub_2620360(v27, v29, v12, v11, v7);
    a4 = v6 - v11;
    a1 = (__int64)v12;
    a3 = v30;
    a5 = v5 - v7;
    a2 = (__int64)v28;
  }
}
