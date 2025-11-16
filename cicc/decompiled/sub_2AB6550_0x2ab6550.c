// Function: sub_2AB6550
// Address: 0x2ab6550
//
__int64 __fastcall sub_2AB6550(
        __int64 a1,
        unsigned __int8 *a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  unsigned __int64 v6; // rcx
  int v8; // esi
  __int64 v9; // r8
  __int64 v10; // rdi
  int v11; // r11d
  unsigned int i; // eax
  __int64 v13; // rdx
  unsigned int v14; // eax
  __int64 result; // rax
  int v16; // eax
  __int64 v17; // rbx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r13
  int v21; // r13d
  __int64 v22; // rax
  __int64 v23; // rdx
  unsigned __int8 *v24; // rax
  __int64 v25; // r8
  __int64 v26; // rdx
  unsigned __int8 *v27; // r13
  __int64 v28; // rbx
  __int64 v29; // r13
  int v30; // edx
  int v31; // ebx
  int v32; // r8d
  int v33; // edx
  bool v34; // cc
  __int64 v35; // rax
  unsigned __int8 *v36; // [rsp+0h] [rbp-80h]
  __int64 v37; // [rsp+10h] [rbp-70h]
  __int64 v38; // [rsp+18h] [rbp-68h]
  _BYTE *v39; // [rsp+20h] [rbp-60h] BYREF
  __int64 v40; // [rsp+28h] [rbp-58h]
  _BYTE v41[80]; // [rsp+30h] [rbp-50h] BYREF

  v6 = HIDWORD(a3);
  v8 = a3;
  HIDWORD(v38) = HIDWORD(a3);
  if ( BYTE4(a3) != 1 && (_DWORD)a3 == 1 )
  {
    v16 = *a2;
    if ( (_BYTE)v16 == 85 )
    {
      v35 = *((_QWORD *)a2 - 4);
      if ( !v35
        || *(_BYTE *)v35
        || *(_QWORD *)(v35 + 24) != *((_QWORD *)a2 + 10)
        || (*(_BYTE *)(v35 + 33) & 0x20) == 0
        || *(_DWORD *)(v35 + 36) != 174 )
      {
        v17 = -32;
        v39 = v41;
        v40 = 0x400000000LL;
        goto LABEL_19;
      }
      sub_2AB5570((__int64)&v39, a1, (__int64)a2, a3, *((_QWORD *)a2 + 1));
      if ( v41[0] )
        return (__int64)v39;
      v16 = *a2;
    }
    v39 = v41;
    v40 = 0x400000000LL;
    if ( v16 == 40 )
    {
      v17 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a2);
    }
    else if ( v16 == 85 )
    {
      v17 = -32;
    }
    else
    {
      v17 = -96;
      if ( v16 != 34 )
        BUG();
    }
LABEL_19:
    if ( (a2[7] & 0x80u) != 0 )
    {
      v18 = sub_BD2BC0((__int64)a2);
      v20 = v18 + v19;
      if ( (a2[7] & 0x80u) == 0 )
      {
        if ( (unsigned int)(v20 >> 4) )
          goto LABEL_52;
      }
      else if ( (unsigned int)((v20 - sub_BD2BC0((__int64)a2)) >> 4) )
      {
        if ( (a2[7] & 0x80u) != 0 )
        {
          v21 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
          if ( (a2[7] & 0x80u) == 0 )
            BUG();
          v22 = sub_BD2BC0((__int64)a2);
          v17 -= 32LL * (unsigned int)(*(_DWORD *)(v22 + v23 - 4) - v21);
          goto LABEL_25;
        }
LABEL_52:
        BUG();
      }
    }
LABEL_25:
    v24 = &a2[v17];
    v25 = (unsigned int)v40;
    v26 = 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
    v27 = &a2[-v26];
    if ( &a2[-v26] != &a2[v17] )
    {
      do
      {
        v28 = *(_QWORD *)(*(_QWORD *)v27 + 8LL);
        if ( v25 + 1 > (unsigned __int64)HIDWORD(v40) )
        {
          v36 = v24;
          sub_C8D5F0((__int64)&v39, v41, v25 + 1, 8u, v25, a6);
          v25 = (unsigned int)v40;
          v24 = v36;
        }
        v27 += 32;
        *(_QWORD *)&v39[8 * v25] = v28;
        v25 = (unsigned int)(v40 + 1);
        LODWORD(v40) = v40 + 1;
      }
      while ( v24 != v27 );
    }
    v29 = sub_DFD7B0(*(_QWORD *)(a1 + 448));
    v31 = v30;
    v32 = sub_9B78C0((__int64)a2, *(__int64 **)(a1 + 456));
    result = v29;
    if ( v32 )
    {
      LODWORD(v38) = 1;
      result = sub_2AB3340(a1, a2, v38);
      v34 = v33 < v31;
      if ( v33 == v31 )
        v34 = result < v29;
      if ( !v34 )
        result = v29;
    }
    if ( v39 != v41 )
    {
      v37 = result;
      _libc_free((unsigned __int64)v39);
      return v37;
    }
    return result;
  }
  v9 = *(unsigned int *)(a1 + 408);
  v10 = *(_QWORD *)(a1 + 392);
  if ( (_DWORD)v9 )
  {
    v11 = 1;
    for ( i = (v9 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)(BYTE4(a3) == 0) + 37 * (_DWORD)a3 - 1)
                | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
             ^ (484763065 * ((BYTE4(a3) == 0) + 37 * a3 - 1))); ; i = (v9 - 1) & v14 )
    {
      v13 = v10 + ((unsigned __int64)i << 6);
      if ( a2 == *(unsigned __int8 **)v13 && v8 == *(_DWORD *)(v13 + 8) && (_BYTE)v6 == *(_BYTE *)(v13 + 12) )
        break;
      if ( *(_QWORD *)v13 == -4096 && *(_DWORD *)(v13 + 8) == -1 && *(_BYTE *)(v13 + 12) )
        goto LABEL_13;
      v14 = v11 + i;
      ++v11;
    }
  }
  else
  {
LABEL_13:
    v13 = v10 + (v9 << 6);
  }
  return *(_QWORD *)(v13 + 48);
}
