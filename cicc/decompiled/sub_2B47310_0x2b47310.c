// Function: sub_2B47310
// Address: 0x2b47310
//
__int64 __fastcall sub_2B47310(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v8; // eax
  unsigned __int64 v9; // rax
  __int64 v10; // rbx
  unsigned __int64 v11; // rdx
  __int64 v12; // rbx
  __int64 i; // rax
  __int64 v14; // rbx
  __int64 result; // rax
  __int64 v16; // r15
  _QWORD *v17; // r12
  __int64 v18; // r13
  __int64 v19; // r14
  unsigned __int64 v20; // rax
  __int64 v21; // rdx
  __int64 j; // rax
  __int64 v23; // r14
  __int64 v24; // rbx
  _QWORD *v25; // r15
  __int64 v26; // rcx
  _BYTE *v27; // rdi
  __int64 v28; // r12
  int v29; // eax
  __int64 v30; // rsi
  int v31; // r10d
  int v32; // eax
  __int64 v33; // rsi
  __int64 v34; // rcx
  __int64 v35; // rsi
  __int64 v36; // rdx
  unsigned __int64 *v37; // rbx
  unsigned __int64 *v38; // r13
  __int64 v39; // [rsp+8h] [rbp-68h]
  unsigned int v41; // [rsp+18h] [rbp-58h]
  unsigned int v42; // [rsp+1Ch] [rbp-54h]
  __int64 v43; // [rsp+20h] [rbp-50h]
  __int64 v44; // [rsp+30h] [rbp-40h]
  unsigned __int64 v45; // [rsp+38h] [rbp-38h]

  v8 = *(_DWORD *)(a4 + 4) & 0x7FFFFFF;
  v41 = v8;
  if ( *(_BYTE *)a4 == 85 )
  {
    v36 = *(_QWORD *)(a4 - 32);
    if ( v36 )
    {
      if ( !*(_BYTE *)v36 && *(_QWORD *)(v36 + 24) == *(_QWORD *)(a4 + 80) )
      {
        v8 = 2;
        if ( (*(_BYTE *)(v36 + 33) & 0x20) == 0 )
          v8 = *(_DWORD *)(a4 + 4) & 0x7FFFFFF;
      }
    }
  }
  *(_DWORD *)(a1 + 208) = v8;
  v9 = *(unsigned int *)(a1 + 8);
  if ( v41 != v9 )
  {
    v10 = 48LL * v41;
    if ( v41 < v9 )
    {
      v37 = (unsigned __int64 *)(*(_QWORD *)a1 + v10);
      v38 = (unsigned __int64 *)(*(_QWORD *)a1 + 48 * v9);
      while ( v37 != v38 )
      {
        v38 -= 6;
        if ( (unsigned __int64 *)*v38 != v38 + 2 )
          _libc_free(*v38);
      }
    }
    else
    {
      v11 = *(unsigned int *)(a1 + 12);
      if ( v41 > v11 )
      {
        sub_2B47150(a1, v41, v11, a4, a5, a6);
        v9 = *(unsigned int *)(a1 + 8);
      }
      v12 = *(_QWORD *)a1 + v10;
      for ( i = *(_QWORD *)a1 + 48 * v9; v12 != i; i += 48 )
      {
        if ( i )
        {
          *(_DWORD *)(i + 8) = 0;
          *(_QWORD *)i = i + 16;
          *(_DWORD *)(i + 12) = 2;
        }
      }
    }
    *(_DWORD *)(a1 + 8) = v41;
  }
  v42 = a3;
  v14 = 0;
  v45 = a3;
  result = 16LL * a3;
  v39 = result;
  if ( v41 )
  {
    v16 = a4;
    v17 = (_QWORD *)a1;
    while ( 1 )
    {
      v18 = 48 * v14;
      v19 = 48 * v14 + *v17;
      v20 = *(unsigned int *)(v19 + 8);
      if ( v45 != v20 )
      {
        if ( v45 >= v20 )
        {
          if ( v45 > *(unsigned int *)(v19 + 12) )
          {
            sub_C8D5F0(48 * v14 + *v17, (const void *)(v19 + 16), v45, 0x10u, a5, a6);
            v20 = *(unsigned int *)(v19 + 8);
          }
          v21 = *(_QWORD *)v19 + v39;
          for ( j = *(_QWORD *)v19 + 16 * v20; v21 != j; j += 16 )
          {
            if ( j )
            {
              *(_QWORD *)j = 0;
              *(_BYTE *)(j + 8) = 0;
              *(_BYTE *)(j + 9) = 0;
            }
          }
        }
        *(_DWORD *)(v19 + 8) = v42;
      }
      v23 = 0;
      result = 32 * v14;
      v43 = 32 * v14;
      if ( v42 )
        break;
LABEL_33:
      if ( v41 == (_DWORD)++v14 )
        return result;
    }
    v44 = v14;
    v24 = v16;
    v25 = v17;
    while ( 1 )
    {
      while ( 1 )
      {
        v27 = *(_BYTE **)(a2 + 8 * v23);
        v28 = 16 * v23;
        if ( *v27 == 13 )
          break;
        LOBYTE(v29) = sub_2B17690((__int64)v27);
        v30 = *(_QWORD *)(a2 + 8 * v23);
        v31 = v29;
        v32 = v44;
        LOBYTE(v32) = (_DWORD)v44 == 0;
        result = (v31 | v32) ^ 1u;
        if ( (*(_BYTE *)(v30 + 7) & 0x40) != 0 )
          v33 = *(_QWORD *)(v30 - 8);
        else
          v33 = v30 - 32LL * (*(_DWORD *)(v30 + 4) & 0x7FFFFFF);
        ++v23;
        v34 = v28 + *(_QWORD *)(*v25 + v18);
        *(_QWORD *)v34 = *(_QWORD *)(v33 + v43);
        *(_BYTE *)(v34 + 8) = result;
        *(_BYTE *)(v34 + 9) = 0;
        if ( v23 == v45 )
        {
LABEL_32:
          v17 = v25;
          v16 = v24;
          v14 = v44;
          goto LABEL_33;
        }
      }
      if ( *(_BYTE *)v24 == 90 )
      {
        if ( v44 )
          goto LABEL_35;
        result = *(_QWORD *)(v24 - 64);
      }
      else
      {
        if ( *(_BYTE *)v24 != 93 || (_DWORD)v44 )
        {
LABEL_35:
          if ( (*(_BYTE *)(v24 + 7) & 0x40) != 0 )
            v35 = *(_QWORD *)(v24 - 8);
          else
            v35 = v24 - 32LL * (*(_DWORD *)(v24 + 4) & 0x7FFFFFF);
          result = sub_ACADE0(*(__int64 ***)(*(_QWORD *)(v35 + v43) + 8LL));
          goto LABEL_27;
        }
        result = *(_QWORD *)(v24 - 32);
      }
LABEL_27:
      ++v23;
      v26 = v28 + *(_QWORD *)(*v25 + v18);
      *(_QWORD *)v26 = result;
      *(_WORD *)(v26 + 8) = 1;
      if ( v23 == v45 )
        goto LABEL_32;
    }
  }
  return result;
}
