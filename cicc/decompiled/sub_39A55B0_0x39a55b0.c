// Function: sub_39A55B0
// Address: 0x39a55b0
//
unsigned __int64 __fastcall sub_39A55B0(__int64 a1, unsigned __int8 *a2, unsigned __int8 *a3)
{
  unsigned int v6; // esi
  __int64 v7; // r8
  unsigned int v8; // edi
  unsigned __int64 result; // rax
  unsigned __int8 *v10; // rcx
  __int64 v11; // rbx
  unsigned int v12; // esi
  __int64 v13; // r8
  unsigned int v14; // edi
  unsigned __int8 *v15; // rcx
  int v16; // r11d
  unsigned __int8 **v17; // rdx
  int v18; // eax
  int v19; // ecx
  int v20; // eax
  int v21; // esi
  __int64 v22; // r8
  int v23; // ecx
  unsigned __int8 *v24; // rdi
  int v25; // r11d
  int v26; // eax
  int v27; // eax
  __int64 v28; // rdi
  unsigned __int8 **v29; // r8
  unsigned int v30; // r14d
  int v31; // r9d
  unsigned __int8 *v32; // rsi
  int v33; // eax
  int v34; // esi
  __int64 v35; // r8
  unsigned __int8 *v36; // rdi
  int v37; // r10d
  unsigned __int8 **v38; // r9
  int v39; // eax
  __int64 v40; // rdi
  unsigned __int8 **v41; // r8
  unsigned int v42; // r14d
  int v43; // r9d
  unsigned __int8 *v44; // rsi
  int v45; // r10d
  unsigned __int8 **v46; // r9

  if ( !(unsigned __int8)sub_39A2350((_QWORD *)a1, a2) )
  {
    v6 = *(_DWORD *)(a1 + 248);
    if ( v6 )
    {
      v7 = *(_QWORD *)(a1 + 232);
      v8 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      result = v7 + 16LL * v8;
      v10 = *(unsigned __int8 **)result;
      if ( a2 == *(unsigned __int8 **)result )
        return result;
      v25 = 1;
      v17 = 0;
      while ( v10 != (unsigned __int8 *)-8LL )
      {
        if ( v17 || v10 != (unsigned __int8 *)-16LL )
          result = (unsigned __int64)v17;
        v8 = (v6 - 1) & (v25 + v8);
        v10 = *(unsigned __int8 **)(v7 + 16LL * v8);
        if ( a2 == v10 )
          return result;
        ++v25;
        v17 = (unsigned __int8 **)result;
        result = v7 + 16LL * v8;
      }
      if ( !v17 )
        v17 = (unsigned __int8 **)result;
      v26 = *(_DWORD *)(a1 + 240);
      ++*(_QWORD *)(a1 + 224);
      v23 = v26 + 1;
      if ( 4 * (v26 + 1) < 3 * v6 )
      {
        result = v6 - *(_DWORD *)(a1 + 244) - v23;
        if ( (unsigned int)result > v6 >> 3 )
        {
LABEL_18:
          *(_DWORD *)(a1 + 240) = v23;
          if ( *v17 != (unsigned __int8 *)-8LL )
            --*(_DWORD *)(a1 + 244);
LABEL_20:
          *v17 = a2;
          v17[1] = a3;
          return result;
        }
        sub_39A53F0(a1 + 224, v6);
        v27 = *(_DWORD *)(a1 + 248);
        if ( v27 )
        {
          result = (unsigned int)(v27 - 1);
          v28 = *(_QWORD *)(a1 + 232);
          v29 = 0;
          v30 = result & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v31 = 1;
          v23 = *(_DWORD *)(a1 + 240) + 1;
          v17 = (unsigned __int8 **)(v28 + 16LL * v30);
          v32 = *v17;
          if ( a2 != *v17 )
          {
            while ( v32 != (unsigned __int8 *)-8LL )
            {
              if ( !v29 && v32 == (unsigned __int8 *)-16LL )
                v29 = v17;
              v30 = result & (v31 + v30);
              v17 = (unsigned __int8 **)(v28 + 16LL * v30);
              v32 = *v17;
              if ( a2 == *v17 )
                goto LABEL_18;
              ++v31;
            }
            if ( v29 )
              v17 = v29;
          }
          goto LABEL_18;
        }
LABEL_83:
        ++*(_DWORD *)(a1 + 240);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 224);
    }
    sub_39A53F0(a1 + 224, 2 * v6);
    v20 = *(_DWORD *)(a1 + 248);
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = *(_QWORD *)(a1 + 232);
      result = (v20 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v23 = *(_DWORD *)(a1 + 240) + 1;
      v17 = (unsigned __int8 **)(v22 + 16 * result);
      v24 = *v17;
      if ( a2 != *v17 )
      {
        v45 = 1;
        v46 = 0;
        while ( v24 != (unsigned __int8 *)-8LL )
        {
          if ( !v46 && v24 == (unsigned __int8 *)-16LL )
            v46 = v17;
          result = v21 & (unsigned int)(v45 + result);
          v17 = (unsigned __int8 **)(v22 + 16LL * (unsigned int)result);
          v24 = *v17;
          if ( a2 == *v17 )
            goto LABEL_18;
          ++v45;
        }
        if ( v46 )
          v17 = v46;
      }
      goto LABEL_18;
    }
    goto LABEL_83;
  }
  v11 = *(_QWORD *)(a1 + 208);
  v12 = *(_DWORD *)(v11 + 384);
  if ( !v12 )
  {
    ++*(_QWORD *)(v11 + 360);
    goto LABEL_34;
  }
  v13 = *(_QWORD *)(v11 + 368);
  v14 = (v12 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  result = v13 + 16LL * v14;
  v15 = *(unsigned __int8 **)result;
  if ( a2 != *(unsigned __int8 **)result )
  {
    v16 = 1;
    v17 = 0;
    while ( v15 != (unsigned __int8 *)-8LL )
    {
      if ( v17 || v15 != (unsigned __int8 *)-16LL )
        result = (unsigned __int64)v17;
      v14 = (v12 - 1) & (v16 + v14);
      v15 = *(unsigned __int8 **)(v13 + 16LL * v14);
      if ( a2 == v15 )
        return result;
      ++v16;
      v17 = (unsigned __int8 **)result;
      result = v13 + 16LL * v14;
    }
    if ( !v17 )
      v17 = (unsigned __int8 **)result;
    v18 = *(_DWORD *)(v11 + 376);
    ++*(_QWORD *)(v11 + 360);
    v19 = v18 + 1;
    if ( 4 * (v18 + 1) < 3 * v12 )
    {
      result = v12 - *(_DWORD *)(v11 + 380) - v19;
      if ( (unsigned int)result > v12 >> 3 )
      {
LABEL_13:
        *(_DWORD *)(v11 + 376) = v19;
        if ( *v17 != (unsigned __int8 *)-8LL )
          --*(_DWORD *)(v11 + 380);
        goto LABEL_20;
      }
      sub_39A53F0(v11 + 360, v12);
      v39 = *(_DWORD *)(v11 + 384);
      if ( v39 )
      {
        result = (unsigned int)(v39 - 1);
        v40 = *(_QWORD *)(v11 + 368);
        v41 = 0;
        v42 = result & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v43 = 1;
        v19 = *(_DWORD *)(v11 + 376) + 1;
        v17 = (unsigned __int8 **)(v40 + 16LL * v42);
        v44 = *v17;
        if ( a2 != *v17 )
        {
          while ( v44 != (unsigned __int8 *)-8LL )
          {
            if ( v44 == (unsigned __int8 *)-16LL && !v41 )
              v41 = v17;
            v42 = result & (v43 + v42);
            v17 = (unsigned __int8 **)(v40 + 16LL * v42);
            v44 = *v17;
            if ( a2 == *v17 )
              goto LABEL_13;
            ++v43;
          }
          if ( v41 )
            v17 = v41;
        }
        goto LABEL_13;
      }
LABEL_82:
      ++*(_DWORD *)(v11 + 376);
      BUG();
    }
LABEL_34:
    sub_39A53F0(v11 + 360, 2 * v12);
    v33 = *(_DWORD *)(v11 + 384);
    if ( v33 )
    {
      v34 = v33 - 1;
      v35 = *(_QWORD *)(v11 + 368);
      result = (v33 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v19 = *(_DWORD *)(v11 + 376) + 1;
      v17 = (unsigned __int8 **)(v35 + 16 * result);
      v36 = *v17;
      if ( a2 != *v17 )
      {
        v37 = 1;
        v38 = 0;
        while ( v36 != (unsigned __int8 *)-8LL )
        {
          if ( v36 == (unsigned __int8 *)-16LL && !v38 )
            v38 = v17;
          result = v34 & (unsigned int)(v37 + result);
          v17 = (unsigned __int8 **)(v35 + 16LL * (unsigned int)result);
          v36 = *v17;
          if ( a2 == *v17 )
            goto LABEL_13;
          ++v37;
        }
        if ( v38 )
          v17 = v38;
      }
      goto LABEL_13;
    }
    goto LABEL_82;
  }
  return result;
}
