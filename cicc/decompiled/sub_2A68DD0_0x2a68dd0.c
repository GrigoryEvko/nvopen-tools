// Function: sub_2A68DD0
// Address: 0x2a68dd0
//
void __fastcall sub_2A68DD0(__int64 a1, __int64 a2, _QWORD *a3)
{
  unsigned __int64 v5; // rbx
  __int64 v6; // r8
  __int64 v7; // r9
  unsigned __int64 v8; // rax
  _BYTE *v9; // rax
  _BYTE *i; // rdx
  unsigned __int8 v11; // al
  unsigned __int8 *v12; // rax
  __int64 v13; // rax
  unsigned int v14; // ebx
  bool v15; // al
  unsigned int v16; // ecx
  _QWORD *v17; // rbx
  _BYTE *v18; // rax
  _BYTE *v19; // rsi
  __int64 v20; // r9
  __int64 v21; // rax
  __int64 v22; // r8
  __int64 v23; // rdi
  unsigned int v24; // ecx
  _BYTE *v25; // rdx
  __int64 v26; // r10
  _BYTE *v27; // rdx
  _BYTE *v28; // rdx
  _BYTE *v29; // rdx
  int v30; // edx
  __int64 v31; // rax
  unsigned int v32; // eax
  __int64 v33; // r8
  __int64 v34; // r9
  size_t v35; // rbx
  size_t v36; // rax
  size_t v37; // rdx
  unsigned __int8 *v38; // rax
  _BYTE *v39; // rax
  __int64 v40; // rdx
  int v41; // esi
  __int64 v42; // rcx
  unsigned int v43; // eax
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 v46; // r9
  void *v47; // rdi
  unsigned __int8 v48; // al
  __int64 v49; // r15
  unsigned int v50; // eax
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // r9
  unsigned int v54; // eax
  __int64 v55; // r14
  __int64 v56; // rbx
  __int64 v57; // rax
  unsigned __int64 v58; // rsi
  __int64 v59; // rdx
  __int64 v60; // rdi
  __int64 v61; // rcx
  _BYTE *v62; // rax
  _BYTE *v63; // rdx
  _BYTE *v64; // rdx
  unsigned int v65; // [rsp+Ch] [rbp-64h]
  unsigned __int8 v66[96]; // [rsp+10h] [rbp-60h] BYREF

  v5 = (unsigned int)sub_B46E30(a2);
  v8 = a3[1];
  if ( v5 != v8 )
  {
    if ( v5 >= v8 )
    {
      if ( v5 > a3[2] )
      {
        sub_C8D290((__int64)a3, a3 + 3, v5, 1u, v6, v7);
        v9 = (_BYTE *)(*a3 + a3[1]);
        for ( i = (_BYTE *)(v5 + *a3); i != v9; ++v9 )
        {
LABEL_5:
          if ( v9 )
            *v9 = 0;
        }
      }
      else
      {
        v9 = (_BYTE *)(*a3 + v8);
        i = (_BYTE *)(v5 + *a3);
        if ( v9 != i )
          goto LABEL_5;
      }
    }
    a3[1] = v5;
  }
  v11 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 == 31 )
  {
    if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) != 1 )
    {
      v12 = (unsigned __int8 *)sub_2A68BC0(a1, *(unsigned __int8 **)(a2 - 96));
      sub_22C05A0((__int64)v66, v12);
      v13 = sub_2A637C0(a1, (__int64)v66, *(_QWORD *)(*(_QWORD *)(a2 - 96) + 8LL));
      if ( v13 && *(_BYTE *)v13 == 17 )
      {
        v14 = *(_DWORD *)(v13 + 32);
        if ( v14 <= 0x40 )
          v15 = *(_QWORD *)(v13 + 24) == 0;
        else
          v15 = v14 == (unsigned int)sub_C444A0(v13 + 24);
        *(_BYTE *)(*a3 + v15) = 1;
      }
      else if ( v66[0] > 1u )
      {
        *(_BYTE *)(*a3 + 1LL) = 1;
        *(_BYTE *)*a3 = 1;
      }
LABEL_16:
      sub_22C0090(v66);
      return;
    }
LABEL_17:
    *(_BYTE *)*a3 = 1;
    return;
  }
  v16 = v11 - 29;
  if ( v16 > 6 )
  {
    if ( (unsigned int)v11 - 37 > 3 )
      goto LABEL_21;
LABEL_40:
    v32 = sub_B46E30(a2);
    v35 = v32;
    if ( (unsigned __int64)v32 > a3[2] )
    {
      a3[1] = 0;
      sub_C8D290((__int64)a3, a3 + 3, v32, 1u, v33, v34);
      if ( v35 )
        memset((void *)*a3, 1, v35);
    }
    else
    {
      v36 = a3[1];
      v37 = v36;
      if ( v35 <= v36 )
        v37 = v35;
      if ( v37 )
      {
        memset((void *)*a3, 1, v37);
        v36 = a3[1];
      }
      if ( v35 > v36 )
      {
        v47 = (void *)(*a3 + v36);
        if ( v47 != (void *)(v35 + *a3) )
          memset(v47, 1, v35 - v36);
      }
    }
    a3[1] = v35;
    return;
  }
  if ( v16 > 4 )
    goto LABEL_40;
LABEL_21:
  if ( v11 != 32 )
  {
    if ( v11 != 33 )
      BUG();
    v38 = (unsigned __int8 *)sub_2A68BC0(a1, **(unsigned __int8 ***)(a2 - 8));
    sub_22C05A0((__int64)v66, v38);
    v39 = (_BYTE *)sub_2A637C0(a1, (__int64)v66, *(_QWORD *)(**(_QWORD **)(a2 - 8) + 8LL));
    if ( v39 && *v39 == 4 )
    {
      v40 = 0;
      v41 = (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) - 1;
      while ( (_DWORD)v40 != v41 )
      {
        v42 = (unsigned int)(v40 + 1);
        if ( *((_QWORD *)v39 - 4) == *(_QWORD *)(*(_QWORD *)(a2 - 8) + 32 * v42) )
        {
          *(_BYTE *)(*a3 + v40) = 1;
          goto LABEL_16;
        }
        v40 = (unsigned int)v42;
      }
    }
    else if ( v66[0] > 1u )
    {
      v43 = sub_B46E30(a2);
      sub_2A65660((__int64)a3, v43, 1u, v44, v45, v46);
    }
    goto LABEL_16;
  }
  if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFFu) >> 1 == 1 )
    goto LABEL_17;
  v17 = sub_2A68BC0(a1, **(unsigned __int8 ***)(a2 - 8));
  v18 = (_BYTE *)sub_2A637C0(a1, (__int64)v17, *(_QWORD *)(**(_QWORD **)(a2 - 8) + 8LL));
  v19 = v18;
  if ( v18 && *v18 == 17 )
  {
    v20 = ((*(_DWORD *)(a2 + 4) & 0x7FFFFFFu) >> 1) - 1;
    v21 = v20 >> 2;
    if ( v20 >> 2 )
    {
      v22 = 4 * v21;
      v23 = *(_QWORD *)(a2 - 8);
      v24 = 2;
      v21 = 0;
      while ( 1 )
      {
        v26 = v21 + 1;
        v29 = *(_BYTE **)(v23 + 32LL * v24);
        if ( v29 )
        {
          if ( v19 == v29 )
            break;
        }
        v25 = *(_BYTE **)(v23 + 32LL * (v24 + 2));
        if ( v25 && v19 == v25 )
        {
LABEL_79:
          v21 = v26;
          break;
        }
        v26 = v21 + 3;
        v27 = *(_BYTE **)(v23 + 32LL * (v24 + 4));
        if ( v27 && v19 == v27 )
        {
          v21 += 2;
          break;
        }
        v21 += 4;
        v28 = *(_BYTE **)(v23 + 32LL * (unsigned int)(2 * v21));
        if ( v28 && v19 == v28 )
          goto LABEL_79;
        v24 += 8;
        if ( v22 == v21 )
        {
          v59 = v20 - v21;
          goto LABEL_81;
        }
      }
LABEL_36:
      v30 = v21;
      if ( v20 != v21 )
      {
LABEL_37:
        v31 = (unsigned int)(v30 + 1);
LABEL_38:
        *(_BYTE *)(*a3 + v31) = 1;
        return;
      }
LABEL_89:
      v31 = 0;
      goto LABEL_38;
    }
    v59 = ((*(_DWORD *)(a2 + 4) & 0x7FFFFFFu) >> 1) - 1;
LABEL_81:
    switch ( v59 )
    {
      case 2LL:
        v60 = *(_QWORD *)(a2 - 8);
        break;
      case 3LL:
        v60 = *(_QWORD *)(a2 - 8);
        v64 = *(_BYTE **)(v60 + 32LL * (unsigned int)(2 * (v21 + 1)));
        if ( v64 && v19 == v64 )
          goto LABEL_36;
        ++v21;
        break;
      case 1LL:
        v60 = *(_QWORD *)(a2 - 8);
        v61 = v21;
        goto LABEL_85;
      default:
        goto LABEL_89;
    }
    v61 = v21 + 1;
    v63 = *(_BYTE **)(v60 + 32LL * (unsigned int)(2 * (v21 + 1)));
    if ( v63 && v19 == v63 )
      goto LABEL_36;
LABEL_85:
    v62 = *(_BYTE **)(v60 + 32LL * (unsigned int)(2 * v61 + 2));
    if ( v62 )
    {
      if ( v19 == v62 && v20 != v61 )
      {
        v30 = v61;
        if ( v61 != 4294967294LL )
          goto LABEL_37;
      }
    }
    goto LABEL_89;
  }
  v48 = *(_BYTE *)v17;
  v49 = (__int64)(v17 + 1);
  if ( *(_BYTE *)v17 != 4 )
  {
    if ( v48 != 5 )
      goto LABEL_68;
    v49 = (__int64)(v17 + 1);
    if ( !sub_9876C0(v17 + 1) )
    {
      v48 = *(_BYTE *)v17;
LABEL_68:
      if ( v48 > 1u )
      {
        v50 = sub_B46E30(a2);
        sub_2A65660((__int64)a3, v50, 1u, v51, v52, v53);
      }
      return;
    }
  }
  v54 = (*(_DWORD *)(a2 + 4) & 0x7FFFFFFu) >> 1;
  v55 = v54 - 1;
  if ( v54 == 1 )
  {
    v58 = 0;
  }
  else
  {
    v65 = 0;
    v56 = 0;
    do
    {
      if ( sub_AB1B10(v49, *(_QWORD *)(*(_QWORD *)(a2 - 8) + 32LL * (unsigned int)(2 * ++v56)) + 24LL) )
      {
        v57 = (unsigned int)v56;
        if ( (_DWORD)v56 == -1 )
          v57 = 0;
        ++v65;
        *(_BYTE *)(*a3 + v57) = 1;
      }
    }
    while ( v55 != v56 );
    v58 = v65;
  }
  *(_BYTE *)*a3 = sub_AB0550(v49, v58);
}
