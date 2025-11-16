// Function: sub_3026630
// Address: 0x3026630
//
void __fastcall sub_3026630(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // r8
  __int64 v6; // r13
  unsigned __int8 v7; // al
  __int64 v8; // r12
  __int64 v9; // rdi
  char v10; // bl
  int v11; // r13d
  __int64 v12; // rax
  __int64 v13; // rdx
  char v14; // r15
  unsigned int v15; // eax
  int v16; // eax
  __int64 v17; // r8
  unsigned int v18; // ebx
  unsigned int v19; // esi
  __int64 v20; // rax
  __int64 v21; // r10
  __int64 v22; // rsi
  __int64 v23; // r15
  __int64 v24; // r15
  __int64 v25; // rcx
  char v26; // r15
  __int64 v27; // rax
  _QWORD *v28; // rdx
  char v29; // al
  unsigned __int64 v30; // rdx
  unsigned int v31; // eax
  __int64 v32; // r8
  __int64 v33; // r10
  __int64 v34; // rdx
  __int64 *v35; // rsi
  __int64 v36; // rbx
  __int64 v37; // r13
  __int64 v38; // rdx
  __int64 v39; // rsi
  __int64 v40; // rax
  __int64 v41; // r15
  unsigned __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rdx
  char v45; // al
  __int64 v46; // rdx
  unsigned int v47; // eax
  int v48; // [rsp+0h] [rbp-80h]
  int v49; // [rsp+4h] [rbp-7Ch]
  __int64 v50; // [rsp+8h] [rbp-78h]
  __int64 v51; // [rsp+10h] [rbp-70h]
  __int64 v52; // [rsp+10h] [rbp-70h]
  __int64 v53; // [rsp+18h] [rbp-68h]
  __int64 v54; // [rsp+18h] [rbp-68h]
  __int64 v55; // [rsp+18h] [rbp-68h]
  __int64 v56; // [rsp+20h] [rbp-60h]
  __int64 v57; // [rsp+20h] [rbp-60h]
  char v58; // [rsp+20h] [rbp-60h]
  int v60; // [rsp+28h] [rbp-58h]
  __int64 v61; // [rsp+28h] [rbp-58h]
  __int64 v62; // [rsp+28h] [rbp-58h]
  __int64 v63; // [rsp+28h] [rbp-58h]
  __int64 v64; // [rsp+28h] [rbp-58h]
  unsigned __int64 v65; // [rsp+30h] [rbp-50h] BYREF
  __int64 v66; // [rsp+38h] [rbp-48h]
  unsigned __int64 v67; // [rsp+40h] [rbp-40h] BYREF
  __int64 v68; // [rsp+48h] [rbp-38h]

  v4 = sub_31DA930();
  v5 = a2;
  v6 = v4;
  v7 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 != 17 )
  {
    if ( v7 == 18 )
    {
      v16 = sub_C33740(*(_QWORD *)(a2 + 24));
      v5 = a2;
      if ( v16 == 128 )
      {
        v35 = (__int64 *)(a2 + 24);
        if ( *(void **)(a2 + 24) == sub_C33340() )
          sub_C3E660((__int64)&v65, (__int64)v35);
        else
          sub_C3A850((__int64)&v65, v35);
        v5 = a2;
        goto LABEL_4;
      }
      v7 = *(_BYTE *)a2;
    }
    if ( (v7 & 0xFD) == 9 )
    {
      if ( (*(_DWORD *)(v5 + 4) & 0x7FFFFFF) != 0 )
      {
        v36 = 0;
        v37 = 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF);
        do
        {
          if ( (*(_BYTE *)(v5 + 7) & 0x40) != 0 )
            v38 = *(_QWORD *)(v5 - 8);
          else
            v38 = v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF);
          v39 = *(_QWORD *)(v38 + v36);
          v36 += 32;
          v64 = v5;
          sub_3026A90(a1, v39, 0, a3);
          v5 = v64;
        }
        while ( v36 != v37 );
      }
    }
    else if ( (unsigned int)v7 - 15 > 1 )
    {
      if ( v7 != 10 )
        BUG();
      v49 = *(_DWORD *)(v5 + 4) & 0x7FFFFFF;
      if ( v49 )
      {
        v21 = 0;
        v48 = (*(_DWORD *)(v5 + 4) & 0x7FFFFFF) - 1;
        v63 = *(_QWORD *)(v5 + 8);
        do
        {
          v23 = 16 * v21 + 24;
          if ( v48 == (_DWORD)v21 )
          {
            v52 = v5;
            v55 = v21;
            v40 = v23 + sub_AE4AC0(v6, v63);
            v41 = *(_QWORD *)v40;
            v58 = *(_BYTE *)(v40 + 8);
            v42 = sub_BDB740(v6, v63);
            v66 = v43;
            v65 = v42;
            v44 = sub_AE4AC0(v6, v63);
            v45 = *(_BYTE *)(v44 + 32);
            v46 = v65 + *(_QWORD *)(v44 + 24);
            if ( v65 )
              v45 = v66;
            v67 = v46 - v41;
            if ( v41 )
              v45 = v58;
            LOBYTE(v68) = v45;
            v47 = sub_CA1930(&v67);
            v33 = v55;
            v32 = v52;
            v34 = v47;
          }
          else
          {
            v50 = v5;
            v51 = v21;
            v54 = 16 * v21;
            v24 = sub_AE4AC0(v6, v63) + v23;
            v25 = *(_QWORD *)v24;
            v26 = *(_BYTE *)(v24 + 8);
            v57 = v25;
            v27 = sub_AE4AC0(v6, v63);
            v28 = (_QWORD *)(v54 + v27 + 40);
            v29 = *(_BYTE *)(v54 + v27 + 48);
            v30 = *v28 - v57;
            if ( v57 )
              v29 = v26;
            v67 = v30;
            LOBYTE(v68) = v29;
            v31 = sub_CA1930(&v67);
            v32 = v50;
            v33 = v51;
            v34 = v31;
          }
          if ( (*(_BYTE *)(v32 + 7) & 0x40) != 0 )
            v22 = *(_QWORD *)(v32 - 8);
          else
            v22 = v32 - 32LL * (*(_DWORD *)(v32 + 4) & 0x7FFFFFF);
          v53 = v32;
          v56 = v33;
          sub_3026A90(a1, *(_QWORD *)(v22 + 32 * v33), v34, a3);
          v5 = v53;
          v21 = v56 + 1;
        }
        while ( v49 != (_DWORD)v56 + 1 );
      }
    }
    else
    {
      v61 = v5;
      if ( (unsigned int)sub_AC5290(v5) )
      {
        v17 = v61;
        v18 = 0;
        while ( 1 )
        {
          v62 = v17;
          if ( (unsigned int)sub_AC5290(v17) <= v18 )
            break;
          v19 = v18++;
          v20 = sub_AD68C0(v62, v19);
          sub_3026A90(a1, v20, 0, a3);
          v17 = v62;
        }
      }
    }
    return;
  }
  LODWORD(v66) = *(_DWORD *)(a2 + 32);
  if ( (unsigned int)v66 > 0x40 )
  {
    sub_C43780((__int64)&v65, (const void **)(a2 + 24));
    v5 = a2;
  }
  else
  {
    v65 = *(_QWORD *)(a2 + 24);
  }
LABEL_4:
  v8 = *(_QWORD *)(v5 + 8);
  v9 = v6;
  v10 = sub_AE5020(v6, v8);
  v11 = 0;
  v12 = sub_9208B0(v9, v8);
  v68 = v13;
  v67 = (((unsigned __int64)(v12 + 7) >> 3) + (1LL << v10) - 1) >> v10 << v10;
  v60 = sub_CA1930(&v67);
  if ( v60 )
  {
    do
    {
      while ( 1 )
      {
        sub_C443A0((__int64)&v67, (__int64)&v65, 8u);
        v14 = v67;
        if ( (unsigned int)v68 > 0x40 )
        {
          v14 = *(_BYTE *)v67;
          j_j___libc_free_0_0(v67);
        }
        *(_BYTE *)(*(_QWORD *)(a3 + 8) + *(unsigned int *)(a3 + 160)) = v14;
        v15 = v66;
        ++*(_DWORD *)(a3 + 160);
        if ( v15 <= 0x40 )
          break;
        ++v11;
        sub_C482E0((__int64)&v65, 8u);
        if ( v60 == v11 )
          goto LABEL_13;
      }
      if ( v15 == 8 )
        v65 = 0;
      else
        v65 >>= 8;
      ++v11;
    }
    while ( v60 != v11 );
  }
LABEL_13:
  if ( (unsigned int)v66 > 0x40 )
  {
    if ( v65 )
      j_j___libc_free_0_0(v65);
  }
}
