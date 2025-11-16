// Function: sub_1CA7E20
// Address: 0x1ca7e20
//
__int64 __fastcall sub_1CA7E20(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rsi
  int v8; // r8d
  __int64 v10; // rdx
  int v11; // r10d
  unsigned int v12; // r15d
  __int64 v13; // rdi
  unsigned int v14; // r11d
  __int64 v15; // rax
  _BYTE *v16; // rcx
  _BYTE *v17; // r9
  __int64 result; // rax
  __int64 v19; // rdx
  bool v20; // cc
  char v21; // cl
  _QWORD *v22; // rbx
  _QWORD *v23; // rdx
  _QWORD *v24; // r13
  _QWORD *v25; // rdi
  _QWORD *v26; // rax
  __int64 v27; // rsi
  __int64 v28; // rcx
  _QWORD *v29; // r14
  __int64 v30; // rax
  _QWORD *v31; // rsi
  _QWORD *v32; // rax
  _QWORD *v33; // rdx
  _BOOL8 v34; // rdi
  _QWORD *v35; // rdi
  _QWORD *v36; // r10
  int v37; // r11d
  __int64 v38; // r9
  int v39; // eax
  int v40; // edx
  int v41; // eax
  int v42; // eax
  __int64 v43; // rsi
  unsigned int v44; // r15d
  _BYTE *v45; // rcx
  int v46; // r8d
  __int64 v47; // rdi
  int v48; // eax
  int v49; // eax
  __int64 v50; // rsi
  int v51; // r8d
  unsigned int v52; // r15d
  _BYTE *v53; // rcx
  __int64 v54; // r10
  int v55; // [rsp+Ch] [rbp-34h]

  v7 = *(unsigned int *)(a3 + 24);
  if ( (_DWORD)v7 )
  {
    v8 = v7 - 1;
    v10 = *(_QWORD *)(a3 + 8);
    v11 = 1;
    v12 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
    LODWORD(v13) = (v7 - 1) & v12;
    v14 = v13;
    v15 = v10 + 16LL * (unsigned int)v13;
    v16 = *(_BYTE **)v15;
    v17 = *(_BYTE **)v15;
    if ( *(_BYTE **)v15 == a2 )
    {
      if ( v15 != v10 + 16 * v7 )
        return *(unsigned int *)(v15 + 8);
      goto LABEL_8;
    }
    while ( 1 )
    {
      if ( v17 == (_BYTE *)-8LL )
        goto LABEL_8;
      v14 = v8 & (v11 + v14);
      v55 = v11 + 1;
      v36 = (_QWORD *)(v10 + 16LL * v14);
      v17 = (_BYTE *)*v36;
      if ( (_BYTE *)*v36 == a2 )
        break;
      v11 = v55;
    }
    if ( v36 != (_QWORD *)(v10 + 16LL * (unsigned int)v7) )
    {
      v37 = 1;
      v38 = 0;
      while ( v16 != (_BYTE *)-8LL )
      {
        if ( v38 || v16 != (_BYTE *)-16LL )
          v15 = v38;
        v13 = v8 & (unsigned int)(v13 + v37);
        v54 = v10 + 16 * v13;
        v16 = *(_BYTE **)v54;
        if ( *(_BYTE **)v54 == a2 )
          return *(unsigned int *)(v54 + 8);
        ++v37;
        v38 = v15;
        v15 = v10 + 16 * v13;
      }
      if ( !v38 )
        v38 = v15;
      v39 = *(_DWORD *)(a3 + 16);
      ++*(_QWORD *)a3;
      v40 = v39 + 1;
      if ( 4 * (v39 + 1) >= (unsigned int)(3 * v7) )
      {
        sub_177C7D0(a3, 2 * v7);
        v41 = *(_DWORD *)(a3 + 24);
        if ( v41 )
        {
          v42 = v41 - 1;
          v43 = *(_QWORD *)(a3 + 8);
          v44 = v42 & v12;
          v40 = *(_DWORD *)(a3 + 16) + 1;
          v38 = v43 + 16LL * v44;
          v45 = *(_BYTE **)v38;
          if ( *(_BYTE **)v38 == a2 )
            goto LABEL_63;
          v46 = 1;
          v47 = 0;
          while ( v45 != (_BYTE *)-8LL )
          {
            if ( !v47 && v45 == (_BYTE *)-16LL )
              v47 = v38;
            v44 = v42 & (v46 + v44);
            v38 = v43 + 16LL * v44;
            v45 = *(_BYTE **)v38;
            if ( *(_BYTE **)v38 == a2 )
              goto LABEL_63;
            ++v46;
          }
LABEL_72:
          if ( v47 )
            v38 = v47;
          goto LABEL_63;
        }
      }
      else
      {
        if ( (int)v7 - *(_DWORD *)(a3 + 20) - v40 > (unsigned int)v7 >> 3 )
        {
LABEL_63:
          *(_DWORD *)(a3 + 16) = v40;
          if ( *(_QWORD *)v38 != -8 )
            --*(_DWORD *)(a3 + 20);
          *(_QWORD *)v38 = a2;
          *(_DWORD *)(v38 + 8) = 0;
          return 0;
        }
        sub_177C7D0(a3, v7);
        v48 = *(_DWORD *)(a3 + 24);
        if ( v48 )
        {
          v49 = v48 - 1;
          v50 = *(_QWORD *)(a3 + 8);
          v51 = 1;
          v52 = v49 & v12;
          v40 = *(_DWORD *)(a3 + 16) + 1;
          v47 = 0;
          v38 = v50 + 16LL * v52;
          v53 = *(_BYTE **)v38;
          if ( *(_BYTE **)v38 == a2 )
            goto LABEL_63;
          while ( v53 != (_BYTE *)-8LL )
          {
            if ( v53 == (_BYTE *)-16LL && !v47 )
              v47 = v38;
            v52 = v49 & (v51 + v52);
            v38 = v50 + 16LL * v52;
            v53 = *(_BYTE **)v38;
            if ( *(_BYTE **)v38 == a2 )
              goto LABEL_63;
            ++v51;
          }
          goto LABEL_72;
        }
      }
      ++*(_DWORD *)(a3 + 16);
      BUG();
    }
  }
LABEL_8:
  v19 = *(_QWORD *)a2;
  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 15 )
  {
    LODWORD(result) = *(_DWORD *)(v19 + 8) >> 8;
    if ( (_DWORD)result )
    {
LABEL_10:
      v20 = (unsigned int)result <= 4;
      if ( (_DWORD)result != 4 )
        goto LABEL_11;
      return 4;
    }
  }
  v21 = a2[16];
  if ( v21 == 17 )
  {
    if ( unk_4FBE1ED && (unsigned __int8)sub_1C2F070(a4) && !(unsigned __int8)sub_15E0450((__int64)a2) )
      return 1;
    if ( (unsigned __int8)sub_15E0450((__int64)a2) && !(unsigned __int8)sub_1C2F070(a4) )
      return 8;
    v22 = *(_QWORD **)(a1 + 8);
    if ( !v22 )
      return 15;
    v23 = (_QWORD *)v22[2];
    v24 = v22 + 1;
    if ( !v23 )
      return 15;
    v25 = v22 + 1;
    v26 = (_QWORD *)v22[2];
    do
    {
      while ( 1 )
      {
        v27 = v26[2];
        v28 = v26[3];
        if ( v26[4] >= (unsigned __int64)a2 )
          break;
        v26 = (_QWORD *)v26[3];
        if ( !v28 )
          goto LABEL_25;
      }
      v25 = v26;
      v26 = (_QWORD *)v26[2];
    }
    while ( v27 );
LABEL_25:
    result = 15;
    if ( v24 != v25 )
    {
      v29 = v22 + 1;
      if ( v25[4] <= (unsigned __int64)a2 )
      {
        do
        {
          if ( v23[4] < (unsigned __int64)a2 )
          {
            v23 = (_QWORD *)v23[3];
          }
          else
          {
            v29 = v23;
            v23 = (_QWORD *)v23[2];
          }
        }
        while ( v23 );
        if ( v24 == v29 || v29[4] > (unsigned __int64)a2 )
        {
          v30 = sub_22077B0(48);
          v31 = v29;
          *(_QWORD *)(v30 + 32) = a2;
          v29 = (_QWORD *)v30;
          *(_DWORD *)(v30 + 40) = 0;
          v32 = sub_1C704D0(v22, v31, (unsigned __int64 *)(v30 + 32));
          if ( v33 )
          {
            v34 = v32 || v24 == v33 || v33[4] > (unsigned __int64)a2;
            sub_220F040(v34, v29, v33, v22 + 1);
            ++v22[5];
          }
          else
          {
            v35 = v29;
            v29 = v32;
            j_j___libc_free_0(v35, 48);
          }
        }
        result = *((unsigned int *)v29 + 10);
        if ( (_DWORD)result != 4 )
        {
          if ( (unsigned int)result > 4 )
          {
LABEL_37:
            if ( (_DWORD)result != 5 )
            {
              if ( (_DWORD)result == 101 )
                return 16;
              return 15;
            }
            return 8;
          }
          if ( (_DWORD)result != 1 )
          {
LABEL_13:
            if ( (_DWORD)result == 3 )
              return 2;
            return 15;
          }
        }
      }
    }
  }
  else
  {
    if ( v21 == 3 )
    {
      LODWORD(result) = *(_DWORD *)(v19 + 8) >> 8;
      v20 = (unsigned int)result <= 4;
      if ( (_DWORD)result != 4 )
      {
LABEL_11:
        if ( v20 )
        {
          if ( (_DWORD)result != 1 )
            goto LABEL_13;
          return 1;
        }
        goto LABEL_37;
      }
      return 4;
    }
    result = 15;
    if ( v21 == 5 )
    {
      LODWORD(result) = sub_1C95850((__int64)a2, a4);
      goto LABEL_10;
    }
  }
  return result;
}
