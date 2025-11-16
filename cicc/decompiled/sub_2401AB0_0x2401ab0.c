// Function: sub_2401AB0
// Address: 0x2401ab0
//
__int64 __fastcall sub_2401AB0(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  int v9; // r12d
  unsigned int v11; // edx
  __int64 v12; // rsi
  __int64 result; // rax
  __int64 *v14; // rcx
  __int64 v15; // r10
  char v16; // si
  int v17; // eax
  __int64 v18; // r8
  int v19; // edx
  _BYTE *v20; // rcx
  int v21; // ecx
  int v22; // eax
  __int64 v23; // rcx
  int v24; // edx
  _BYTE *v25; // rsi
  int v26; // r8d
  int v27; // eax
  __int64 v28; // rcx
  int v29; // edx
  _BYTE *v30; // rsi
  int v31; // r8d
  _QWORD *v32; // rdi
  __int64 v33; // r9
  __int64 v34; // rdx
  __int64 *v35; // rax
  __int64 *v36; // r15
  __int64 v37; // rdi
  int v38; // r10d
  int v39; // r11d
  __int64 v40; // [rsp+0h] [rbp-80h]
  __int64 v42; // [rsp+8h] [rbp-78h]
  _BYTE *v43; // [rsp+18h] [rbp-68h] BYREF
  _BYTE v44[96]; // [rsp+20h] [rbp-60h] BYREF

  v9 = a4;
  v11 = *(_DWORD *)(a4 + 24);
  v12 = *(_QWORD *)(a4 + 8);
  if ( !v11 )
  {
LABEL_10:
    result = 5LL * v11;
    v14 = (__int64 *)(v12 + 40LL * v11);
    goto LABEL_3;
  }
  result = (v11 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v14 = (__int64 *)(v12 + 40 * result);
  v15 = *v14;
  if ( a3 != *v14 )
  {
    v21 = 1;
    while ( v15 != -4096 )
    {
      v39 = v21 + 1;
      result = (v11 - 1) & (v21 + (_DWORD)result);
      v14 = (__int64 *)(v12 + 40LL * (unsigned int)result);
      v15 = *v14;
      if ( a3 == *v14 )
        goto LABEL_3;
      v21 = v39;
    }
    goto LABEL_10;
  }
LABEL_3:
  v16 = *a1;
  if ( *a1 > 0x1Cu )
  {
    v43 = a1;
    if ( a1 != (_BYTE *)a2 )
    {
      v17 = *((_DWORD *)v14 + 8);
      v18 = v14[2];
      if ( v17 )
      {
        v19 = v17 - 1;
        result = (v17 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
        v20 = *(_BYTE **)(v18 + 8 * result);
        if ( a1 == v20 )
          return result;
        v38 = 1;
        while ( v20 != (_BYTE *)-4096LL )
        {
          result = v19 & (unsigned int)(v38 + result);
          v20 = *(_BYTE **)(v18 + 8LL * (unsigned int)result);
          if ( a1 == v20 )
            return result;
          ++v38;
        }
      }
      if ( v16 == 84 )
      {
        v22 = *(_DWORD *)(a6 + 24);
        v23 = *(_QWORD *)(a6 + 8);
        if ( v22 )
        {
          v24 = v22 - 1;
          result = (v22 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
          v25 = *(_BYTE **)(v23 + 8 * result);
          if ( a1 == v25 )
            return result;
          v26 = 1;
          while ( v25 != (_BYTE *)-4096LL )
          {
            result = v24 & (unsigned int)(v26 + result);
            v25 = *(_BYTE **)(v23 + 8LL * (unsigned int)result);
            if ( a1 == v25 )
              return result;
            ++v26;
          }
        }
      }
      v27 = *(_DWORD *)(a5 + 24);
      v28 = *(_QWORD *)(a5 + 8);
      if ( v27 )
      {
        v29 = v27 - 1;
        result = (v27 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
        v30 = *(_BYTE **)(v28 + 8 * result);
        if ( a1 == v30 )
          return result;
        v31 = 1;
        while ( v30 != (_BYTE *)-4096LL )
        {
          result = v29 & (unsigned int)(v31 + result);
          v30 = *(_BYTE **)(v28 + 8LL * (unsigned int)result);
          if ( a1 == v30 )
            return result;
          ++v31;
        }
      }
      result = sub_B19DB0(a7, (__int64)a1, a2);
      if ( !(_BYTE)result )
      {
        v32 = v43;
        v33 = a6;
        v34 = 4LL * (*((_DWORD *)v43 + 1) & 0x7FFFFFF);
        if ( (v43[7] & 0x40) != 0 )
        {
          v35 = (__int64 *)*((_QWORD *)v43 - 1);
          v40 = (__int64)&v35[v34];
        }
        else
        {
          v40 = (__int64)v43;
          v35 = (__int64 *)&v43[-(v34 * 8)];
        }
        if ( (__int64 *)v40 != v35 )
        {
          v36 = v35;
          do
          {
            v37 = *v36;
            v42 = v33;
            v36 += 4;
            sub_2401AB0(v37, a2, a3, v9, a5, v33, a7);
            v33 = v42;
          }
          while ( (__int64 *)v40 != v36 );
          v32 = v43;
        }
        sub_B444E0(v32, a2 + 24, 0);
        return sub_2400480((__int64)v44, a5, (__int64 *)&v43);
      }
    }
  }
  return result;
}
