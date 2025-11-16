// Function: sub_28DE710
// Address: 0x28de710
//
void __fastcall sub_28DE710(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rdx
  unsigned __int64 v7; // rax
  __int64 v8; // r15
  unsigned int i; // ebx
  unsigned int v10; // esi
  __int64 v11; // rax
  _BYTE *v12; // rax
  _BYTE *v13; // r13
  _BYTE *v14; // rdx
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  __int64 v17; // r15
  int v18; // r14d
  unsigned int j; // ebx
  unsigned int v20; // esi
  __int64 v21; // rax
  __int64 *v22; // r15
  __int64 v23; // rbx
  __int64 v24; // rcx
  __int64 v25; // r14
  char v26; // al
  unsigned int v27; // edx
  __int64 v28; // rdi
  int v29; // eax
  bool v30; // al
  bool v31; // al
  __int64 v32; // rdx
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rdi
  __int64 v36; // r8
  __int64 v37; // rax
  __int64 v38; // r9
  unsigned int v39; // esi
  _BYTE *v40; // rcx
  __int64 v41; // r10
  _BYTE *v42; // rcx
  _BYTE *v43; // rcx
  _BYTE *v44; // rcx
  int v45; // edx
  __int64 v46; // rax
  __int64 v47; // rcx
  _BYTE *v48; // rcx
  _BYTE *v49; // rcx
  _BYTE *v50; // rcx
  unsigned int v51; // [rsp+Ch] [rbp-A4h]
  int v52; // [rsp+18h] [rbp-98h]
  __int64 v53; // [rsp+18h] [rbp-98h]
  __int64 v54[7]; // [rsp+20h] [rbp-90h] BYREF
  int v55; // [rsp+58h] [rbp-58h]
  char v56; // [rsp+5Ch] [rbp-54h]
  char v57; // [rsp+60h] [rbp-50h] BYREF

  if ( *(_BYTE *)a2 != 31 )
  {
    if ( *(_BYTE *)a2 != 32 )
    {
LABEL_3:
      v6 = *(_QWORD *)(a2 + 40);
      v7 = *(_QWORD *)(v6 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v7 == v6 + 48 )
      {
LABEL_9:
        v12 = (_BYTE *)sub_28C8480(a1, a2);
        v13 = v12;
        if ( v12 && *v12 != 26 )
        {
          v33 = sub_28C7B90(a1, (__int64)v12);
          if ( v13 != *(_BYTE **)(v33 + 48) )
          {
            v34 = sub_28CC470(a1, 0, 0);
            *(_QWORD *)(v34 + 48) = v13;
            v33 = v34;
          }
          if ( (unsigned __int8)sub_28CC2D0(a1, v13, v33) )
            sub_28CA760(a1, (__int64)v13);
        }
        return;
      }
      if ( v7 )
      {
        v8 = v7 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v7 - 24) - 30 <= 0xA )
        {
          v52 = sub_B46E30(v8);
          if ( v52 )
          {
            for ( i = 0; i != v52; ++i )
            {
              v10 = i;
              v11 = sub_B46EC0(v8, v10);
              sub_28DC070(a1, a3, v11);
            }
          }
        }
        goto LABEL_9;
      }
LABEL_42:
      BUG();
    }
    v14 = (_BYTE *)sub_28C86C0(a1, **(_QWORD **)(a2 - 8));
    if ( *v14 != 17 )
    {
      v15 = *(_QWORD *)(a2 + 40);
      v16 = *(_QWORD *)(v15 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v16 == v15 + 48 )
        return;
      if ( v16 )
      {
        v17 = v16 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v16 - 24) - 30 <= 0xA )
        {
          v18 = sub_B46E30(v17);
          if ( v18 )
          {
            for ( j = 0; j != v18; ++j )
            {
              v20 = j;
              v21 = sub_B46EC0(v17, v20);
              sub_28DC070(a1, a3, v21);
            }
          }
        }
        return;
      }
      goto LABEL_42;
    }
    v35 = *(_QWORD *)(a2 - 8);
    v36 = ((*(_DWORD *)(a2 + 4) & 0x7FFFFFFu) >> 1) - 1;
    v37 = v36 >> 2;
    if ( v36 >> 2 )
    {
      v38 = 4 * v37;
      v39 = 2;
      v37 = 0;
      while ( 1 )
      {
        v41 = v37 + 1;
        v44 = *(_BYTE **)(v35 + 32LL * v39);
        if ( v44 )
        {
          if ( v14 == v44 )
            break;
        }
        v40 = *(_BYTE **)(v35 + 32LL * (v39 + 2));
        if ( v40 && v14 == v40 )
        {
LABEL_65:
          v37 = v41;
          break;
        }
        v41 = v37 + 3;
        v42 = *(_BYTE **)(v35 + 32LL * (v39 + 4));
        if ( v42 && v14 == v42 )
        {
          v37 += 2;
          break;
        }
        v37 += 4;
        v43 = *(_BYTE **)(v35 + 32LL * (unsigned int)(2 * v37));
        if ( v43 && v14 == v43 )
          goto LABEL_65;
        v39 += 8;
        if ( v37 == v38 )
        {
          v47 = v36 - v37;
          goto LABEL_67;
        }
      }
LABEL_54:
      v45 = v37;
      if ( v37 != v36 )
      {
LABEL_55:
        v46 = 32LL * (unsigned int)(2 * v45 + 3);
        goto LABEL_56;
      }
LABEL_74:
      v46 = 32;
LABEL_56:
      v32 = *(_QWORD *)(v35 + v46);
      if ( *(_QWORD *)(v35 + 32) == v32 )
        v32 = *(_QWORD *)(v35 + 32);
      goto LABEL_33;
    }
    v47 = ((*(_DWORD *)(a2 + 4) & 0x7FFFFFFu) >> 1) - 1;
LABEL_67:
    if ( v47 != 2 )
    {
      if ( v47 != 3 )
      {
        if ( v47 != 1 )
          goto LABEL_74;
        goto LABEL_70;
      }
      v49 = *(_BYTE **)(v35 + 32LL * (unsigned int)(2 * (v37 + 1)));
      if ( v49 && v14 == v49 )
        goto LABEL_54;
      ++v37;
    }
    v50 = *(_BYTE **)(v35 + 32LL * (unsigned int)(2 * (v37 + 1)));
    if ( v50 && v14 == v50 )
      goto LABEL_54;
    ++v37;
LABEL_70:
    v48 = *(_BYTE **)(v35 + 32LL * (unsigned int)(2 * v37 + 2));
    if ( v48 )
    {
      if ( v14 == v48 && v36 != v37 )
      {
        v45 = v37;
        if ( v37 != 4294967294LL )
          goto LABEL_55;
      }
    }
    goto LABEL_74;
  }
  if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) != 3 )
    goto LABEL_3;
  v22 = *(__int64 **)(a2 - 96);
  if ( !v22 )
    goto LABEL_3;
  v53 = *(_QWORD *)(a2 - 32);
  if ( !v53 )
    goto LABEL_3;
  v23 = *(_QWORD *)(a2 - 64);
  if ( !v23 )
    goto LABEL_3;
  v25 = sub_28C86C0(a1, (__int64)v22);
  v26 = *(_BYTE *)v25;
  if ( *(_BYTE *)v25 <= 0x15u )
  {
LABEL_25:
    if ( v26 == 17 )
      goto LABEL_26;
LABEL_41:
    sub_28DC070(a1, a3, v53);
    v32 = v23;
    goto LABEL_33;
  }
  if ( *(_BYTE *)v22 > 0x1Cu )
  {
    v56 = 1;
    v54[5] = (__int64)&v57;
    v54[4] = 0;
    v54[6] = 4;
    v55 = 0;
    sub_28DCC60(v54, a1, (__int64)v22, v24, (__int64)v54);
    if ( !v54[0] )
      goto LABEL_41;
    if ( *(_DWORD *)(v54[0] + 8) != 1 )
      goto LABEL_41;
    v25 = *(_QWORD *)(v54[0] + 24);
    sub_28D37B0(a1, (__int64)v54, v22);
    if ( !v25 )
      goto LABEL_41;
    v26 = *(_BYTE *)v25;
    goto LABEL_25;
  }
  if ( *(_BYTE *)v22 != 17 )
    goto LABEL_41;
  v25 = (__int64)v22;
LABEL_26:
  v27 = *(_DWORD *)(v25 + 32);
  v28 = v25 + 24;
  if ( v27 <= 0x40 )
  {
    v30 = *(_QWORD *)(v25 + 24) == 1;
  }
  else
  {
    v51 = *(_DWORD *)(v25 + 32);
    v29 = sub_C444A0(v28);
    v27 = v51;
    v28 = v25 + 24;
    v30 = v51 - 1 == v29;
  }
  if ( v30 )
  {
    v32 = v53;
    goto LABEL_33;
  }
  if ( v27 <= 0x40 )
    v31 = *(_QWORD *)(v25 + 24) == 0;
  else
    v31 = v27 == (unsigned int)sub_C444A0(v28);
  if ( v31 )
  {
    v32 = v23;
LABEL_33:
    sub_28DC070(a1, a3, v32);
  }
}
