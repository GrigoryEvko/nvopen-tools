// Function: sub_D497B0
// Address: 0xd497b0
//
__int64 __fastcall sub_D497B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r12d
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rsi
  __int64 v13; // r15
  _BYTE *v14; // rax
  unsigned __int8 v15; // dl
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 *v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 *v22; // r14
  __int64 *v23; // r13
  __int64 *v24; // rax
  __int64 v25; // rax
  _QWORD *v26; // rax
  __int64 v28; // rbx
  __int64 v29; // r12
  __int64 v30; // rax
  __int64 v31; // rcx
  __int64 v32; // rdx
  __int64 v33; // rax
  unsigned __int8 v34; // al
  __int64 v35; // rdx
  __int64 v36; // rdx
  _QWORD *v37; // rdi
  __int64 v38; // rdx
  _QWORD *v39; // rdx
  __int64 *v40; // r8
  __int64 *v41; // r13
  __int64 *v42; // rax
  __int64 *v43; // rdx
  __int64 v44; // [rsp+0h] [rbp-90h]
  __int64 v45; // [rsp+8h] [rbp-88h]
  __int64 v46; // [rsp+10h] [rbp-80h]
  __int64 *v47; // [rsp+18h] [rbp-78h]
  __int64 v48; // [rsp+20h] [rbp-70h] BYREF
  __int64 *v49; // [rsp+28h] [rbp-68h]
  __int64 v50; // [rsp+30h] [rbp-60h]
  int v51; // [rsp+38h] [rbp-58h]
  unsigned __int8 v52; // [rsp+3Ch] [rbp-54h]
  char v53; // [rsp+40h] [rbp-50h] BYREF

  v6 = 0;
  v8 = sub_D49300(a1, a2, a3, a4, a5, a6);
  if ( !v8 )
    return v6;
  v12 = (__int64)"llvm.loop.parallel_accesses";
  v13 = v8;
  v14 = sub_D49780(a1, "llvm.loop.parallel_accesses", 0x1Bu, v9, v10, v11);
  v52 = 1;
  v48 = 0;
  v49 = (__int64 *)&v53;
  v50 = 4;
  v51 = 0;
  if ( !v14 )
  {
    v25 = *(_QWORD *)(a1 + 32);
    v44 = *(_QWORD *)(a1 + 40);
    if ( v44 == v25 )
      return 1;
    goto LABEL_27;
  }
  v15 = *(v14 - 16);
  if ( (v15 & 2) != 0 )
  {
    v16 = *((_QWORD *)v14 - 4);
    v12 = *((unsigned int *)v14 - 6);
  }
  else
  {
    v12 = (*((_WORD *)v14 - 8) >> 6) & 0xF;
    v16 = (__int64)&v14[-8 * ((v15 >> 2) & 0xF) - 16];
  }
  v17 = sub_D46550(v16, v12, 1);
  v22 = v18;
  v23 = (__int64 *)v17;
  if ( (__int64 *)v17 != v18 )
  {
    do
    {
      v12 = *v23;
      if ( !(_BYTE)v20 )
        goto LABEL_72;
      v24 = v49;
      v18 = &v49[HIDWORD(v50)];
      if ( v49 != v18 )
      {
        while ( v12 != *v24 )
        {
          if ( v18 == ++v24 )
            goto LABEL_73;
        }
        goto LABEL_11;
      }
LABEL_73:
      if ( HIDWORD(v50) < (unsigned int)v50 )
      {
        ++HIDWORD(v50);
        *v18 = v12;
        v20 = v52;
        ++v48;
      }
      else
      {
LABEL_72:
        sub_C8CC70((__int64)&v48, v12, (__int64)v18, v19, v20, v21);
        v20 = v52;
      }
LABEL_11:
      ++v23;
    }
    while ( v22 != v23 );
  }
  v25 = *(_QWORD *)(a1 + 32);
  v44 = *(_QWORD *)(a1 + 40);
  if ( v25 != v44 )
  {
LABEL_27:
    v45 = v25;
    while ( 1 )
    {
      v28 = *(_QWORD *)(*(_QWORD *)v45 + 56LL);
      v46 = *(_QWORD *)v45 + 48LL;
      if ( v46 != v28 )
        break;
LABEL_51:
      v45 += 8;
      if ( v44 == v45 )
        goto LABEL_13;
    }
    while ( 1 )
    {
      v29 = v28 - 24;
      if ( !v28 )
        v29 = 0;
      if ( !(unsigned __int8)sub_B46420(v29) && !(unsigned __int8)sub_B46490(v29) )
        goto LABEL_50;
      if ( (*(_BYTE *)(v29 + 7) & 0x20) == 0 )
        goto LABEL_22;
      v30 = sub_B91C10(v29, 25);
      v12 = v30;
      if ( !v30 )
        goto LABEL_38;
      v32 = *(unsigned __int8 *)(v30 - 16);
      if ( (v32 & 2) != 0 )
      {
        v30 = *(unsigned int *)(v30 - 24);
        if ( !(_DWORD)v30 )
          goto LABEL_37;
        v40 = *(__int64 **)(v12 - 32);
      }
      else
      {
        LODWORD(v30) = (*(_WORD *)(v30 - 16) >> 6) & 0xF;
        if ( !(_DWORD)v30 )
        {
LABEL_37:
          if ( !(unsigned __int8)sub_B19060((__int64)&v48, v12, v32, v31) )
            goto LABEL_38;
          goto LABEL_50;
        }
        v30 = (unsigned __int8)v30;
        v40 = (__int64 *)(v12 + -16 - 8LL * (((unsigned __int8)v32 >> 2) & 0xF));
      }
      v41 = v40;
      v47 = &v40[v30];
      v12 = *v40;
      if ( v52 )
      {
LABEL_57:
        v42 = v49;
        v43 = &v49[HIDWORD(v50)];
        if ( v49 != v43 )
        {
          while ( v12 != *v42 )
          {
            if ( v43 == ++v42 )
              goto LABEL_60;
          }
          goto LABEL_50;
        }
LABEL_60:
        if ( v47 != ++v41 )
          goto LABEL_61;
LABEL_38:
        if ( (*(_BYTE *)(v29 + 7) & 0x20) == 0 || (v33 = sub_B91C10(v29, 10), (v12 = v33) == 0) )
        {
LABEL_22:
          v6 = 0;
          goto LABEL_23;
        }
        v34 = *(_BYTE *)(v33 - 16);
        if ( (v34 & 2) != 0 )
        {
          v26 = *(_QWORD **)(v12 - 32);
          v35 = *(unsigned int *)(v12 - 24);
        }
        else
        {
          v35 = (*(_WORD *)(v12 - 16) >> 6) & 0xF;
          v26 = (_QWORD *)(v12 + -16 - 8LL * ((v34 >> 2) & 0xF));
        }
        v36 = 8 * v35;
        v37 = &v26[(unsigned __int64)v36 / 8];
        v12 = v36 >> 3;
        v38 = v36 >> 5;
        if ( v38 )
        {
          v39 = &v26[4 * v38];
          while ( v13 != *v26 )
          {
            if ( v13 == v26[1] )
            {
              ++v26;
              goto LABEL_49;
            }
            if ( v13 == v26[2] )
            {
              v26 += 2;
              goto LABEL_49;
            }
            if ( v13 == v26[3] )
            {
              v26 += 3;
              goto LABEL_49;
            }
            v26 += 4;
            if ( v39 == v26 )
            {
              v12 = v37 - v26;
              goto LABEL_15;
            }
          }
          goto LABEL_49;
        }
LABEL_15:
        switch ( v12 )
        {
          case 2LL:
LABEL_19:
            if ( v13 == *v26 )
              goto LABEL_49;
            ++v26;
            break;
          case 3LL:
            if ( v13 == *v26 )
              goto LABEL_49;
            ++v26;
            goto LABEL_19;
          case 1LL:
            break;
          default:
            goto LABEL_22;
        }
        if ( v13 != *v26 )
          goto LABEL_22;
LABEL_49:
        if ( v37 == v26 )
          goto LABEL_22;
        goto LABEL_50;
      }
      while ( !sub_C8CA60((__int64)&v48, v12) )
      {
        if ( v47 == ++v41 )
          goto LABEL_38;
LABEL_61:
        v12 = *v41;
        if ( v52 )
          goto LABEL_57;
      }
LABEL_50:
      v28 = *(_QWORD *)(v28 + 8);
      if ( v46 == v28 )
        goto LABEL_51;
    }
  }
LABEL_13:
  v6 = 1;
LABEL_23:
  if ( !v52 )
    _libc_free(v49, v12);
  return v6;
}
