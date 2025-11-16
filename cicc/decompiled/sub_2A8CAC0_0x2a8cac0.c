// Function: sub_2A8CAC0
// Address: 0x2a8cac0
//
void __fastcall sub_2A8CAC0(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 *v5; // r14
  unsigned __int64 v6; // rbx
  bool v7; // al
  __int64 v8; // rsi
  int v9; // eax
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rdi
  __int64 v13; // rsi
  __int64 *v14; // rbx
  unsigned __int64 v15; // r15
  __int64 *v16; // r13
  __int64 v17; // rsi
  int v18; // eax
  __int64 v19; // rsi
  __int64 v20; // rcx
  int v21; // eax
  __int64 v22; // rdx
  int v23; // ecx
  __int64 v24; // rbx
  unsigned __int64 v25; // r15
  unsigned __int64 v26; // rdx
  unsigned int v27; // ecx
  __int64 v28; // rsi
  unsigned int v29; // edx
  __int64 v30; // rcx
  __int64 v31; // rsi
  unsigned __int64 v32; // rax
  __int64 v33; // rsi
  __int64 *v34; // [rsp+0h] [rbp-60h]
  __int64 v35; // [rsp+8h] [rbp-58h]
  __int64 v36; // [rsp+8h] [rbp-58h]
  __int64 v37; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int64 v38; // [rsp+18h] [rbp-48h]
  unsigned int v39; // [rsp+20h] [rbp-40h]

  v3 = (__int64)a2 - a1;
  v35 = a3;
  if ( (__int64)a2 - a1 <= 384 )
    return;
  v5 = a2;
  if ( !a3 )
  {
    v16 = a2;
    goto LABEL_19;
  }
  v34 = (__int64 *)(a1 + 24);
  while ( 2 )
  {
    --v35;
    v6 = a1
       + 8
       * (((__int64)(0xAAAAAAAAAAAAAAABLL * (v3 >> 3)) >> 1)
        + ((0xAAAAAAAAAAAAAAABLL * (v3 >> 3)) & 0xFFFFFFFFFFFFFFFELL));
    v7 = sub_B445A0(*(_QWORD *)(a1 + 24), *(_QWORD *)v6);
    v8 = *(v5 - 3);
    if ( !v7 )
    {
      if ( sub_B445A0(*(_QWORD *)(a1 + 24), v8) )
      {
        v12 = *(_QWORD *)a1;
        v21 = *(_DWORD *)(a1 + 16);
        v22 = *(_QWORD *)(a1 + 8);
        v13 = *(_QWORD *)(a1 + 24);
        *(_QWORD *)(a1 + 8) = *(_QWORD *)(a1 + 32);
        v23 = *(_DWORD *)(a1 + 40);
        *(_QWORD *)(a1 + 24) = v12;
        *(_QWORD *)a1 = v13;
        *(_DWORD *)(a1 + 16) = v23;
        *(_QWORD *)(a1 + 32) = v22;
        *(_DWORD *)(a1 + 40) = v21;
        goto LABEL_7;
      }
      if ( !sub_B445A0(*(_QWORD *)v6, *(v5 - 3)) )
      {
        sub_2A8AEC0((__int64 *)a1, (__int64 *)v6);
        v12 = *(_QWORD *)(a1 + 24);
        v13 = *(_QWORD *)a1;
        goto LABEL_7;
      }
LABEL_31:
      sub_2A8AEC0((__int64 *)a1, v5 - 3);
      v12 = *(_QWORD *)(a1 + 24);
      v13 = *(_QWORD *)a1;
      goto LABEL_7;
    }
    if ( !sub_B445A0(*(_QWORD *)v6, v8) )
    {
      if ( !sub_B445A0(*(_QWORD *)(a1 + 24), *(v5 - 3)) )
      {
        sub_2A8AEC0((__int64 *)a1, v34);
        v12 = *(_QWORD *)(a1 + 24);
        v13 = *(_QWORD *)a1;
        goto LABEL_7;
      }
      goto LABEL_31;
    }
    v9 = *(_DWORD *)(a1 + 16);
    v10 = *(_QWORD *)a1;
    *(_DWORD *)(a1 + 16) = 0;
    v11 = *(_QWORD *)(a1 + 8);
    *(_QWORD *)a1 = *(_QWORD *)v6;
    *(_QWORD *)(a1 + 8) = *(_QWORD *)(v6 + 8);
    *(_DWORD *)(a1 + 16) = *(_DWORD *)(v6 + 16);
    *(_QWORD *)v6 = v10;
    *(_QWORD *)(v6 + 8) = v11;
    *(_DWORD *)(v6 + 16) = v9;
    v12 = *(_QWORD *)(a1 + 24);
    v13 = *(_QWORD *)a1;
LABEL_7:
    v14 = v34;
    v15 = (unsigned __int64)v5;
    while ( 1 )
    {
      v16 = v14;
      if ( sub_B445A0(v12, v13) )
        goto LABEL_8;
      do
      {
        v17 = *(_QWORD *)(v15 - 24);
        v15 -= 24LL;
      }
      while ( sub_B445A0(*(_QWORD *)a1, v17) );
      if ( (unsigned __int64)v14 >= v15 )
        break;
      v18 = *((_DWORD *)v14 + 4);
      v19 = *v14;
      *((_DWORD *)v14 + 4) = 0;
      v20 = v14[1];
      *v14 = *(_QWORD *)v15;
      v14[1] = *(_QWORD *)(v15 + 8);
      *((_DWORD *)v14 + 4) = *(_DWORD *)(v15 + 16);
      *(_QWORD *)v15 = v19;
      *(_QWORD *)(v15 + 8) = v20;
      *(_DWORD *)(v15 + 16) = v18;
LABEL_8:
      v13 = *(_QWORD *)a1;
      v12 = v14[3];
      v14 += 3;
    }
    sub_2A8CAC0(v14, v5, v35);
    v3 = (__int64)v14 - a1;
    if ( (__int64)v14 - a1 > 384 )
    {
      if ( v35 )
      {
        v5 = v14;
        continue;
      }
LABEL_19:
      v36 = 0xAAAAAAAAAAAAAAABLL * (v3 >> 3);
      v24 = (v36 - 2) >> 1;
      v25 = a1 + 8 * (v24 + ((v36 - 2) & 0xFFFFFFFFFFFFFFFELL));
      while ( 1 )
      {
        v26 = *(_QWORD *)(v25 + 8);
        v27 = *(_DWORD *)(v25 + 16);
        *(_DWORD *)(v25 + 16) = 0;
        v28 = *(_QWORD *)v25;
        v38 = v26;
        v37 = v28;
        v39 = v27;
        sub_2A8C1E0(a1, v24, v36, (__int64)&v37);
        if ( v39 > 0x40 && v38 )
          j_j___libc_free_0_0(v38);
        v25 -= 24LL;
        if ( !v24 )
          break;
        --v24;
      }
      do
      {
        v29 = *((_DWORD *)v16 - 2);
        v16 -= 3;
        v30 = *v16;
        *((_DWORD *)v16 + 4) = 0;
        v31 = *(_QWORD *)a1;
        v32 = v16[1];
        v39 = v29;
        *v16 = v31;
        v33 = *(_QWORD *)(a1 + 8);
        v37 = v30;
        v16[1] = v33;
        LODWORD(v33) = *(_DWORD *)(a1 + 16);
        v38 = v32;
        *((_DWORD *)v16 + 4) = v33;
        *(_DWORD *)(a1 + 16) = 0;
        sub_2A8C1E0(a1, 0, 0xAAAAAAAAAAAAAAABLL * (((__int64)v16 - a1) >> 3), (__int64)&v37);
        if ( v39 > 0x40 )
        {
          if ( v38 )
            j_j___libc_free_0_0(v38);
        }
      }
      while ( (__int64)v16 - a1 > 24 );
    }
    break;
  }
}
