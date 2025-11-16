// Function: sub_2E89F40
// Address: 0x2e89f40
//
__int64 __fastcall sub_2E89F40(__int64 a1, unsigned int a2)
{
  __int64 v3; // r15
  __int64 v4; // rax
  unsigned int v5; // esi
  int v7; // edx
  int v8; // edx
  unsigned int v9; // ebx
  __int64 v10; // rax
  int v11; // r9d
  _BYTE *v12; // r10
  __int64 v13; // r9
  __int64 v14; // r15
  __int64 v15; // rax
  _BYTE *v16; // rdx
  __int64 v17; // r8
  unsigned int v18; // r13d
  unsigned int j; // ecx
  unsigned __int64 v20; // rdx
  unsigned int v21; // r12d
  __int64 v22; // rax
  int v23; // edx
  unsigned __int64 v24; // r11
  unsigned int v25; // eax
  int v26; // r15d
  int v27; // r13d
  unsigned int i; // r12d
  unsigned int v29; // [rsp+8h] [rbp-88h]
  unsigned int v30; // [rsp+Ch] [rbp-84h]
  _BYTE *v31; // [rsp+10h] [rbp-80h]
  unsigned int v32; // [rsp+18h] [rbp-78h]
  __int64 v33; // [rsp+20h] [rbp-70h] BYREF
  int v34; // [rsp+28h] [rbp-68h]
  _BYTE *v35; // [rsp+30h] [rbp-60h] BYREF
  __int64 v36; // [rsp+38h] [rbp-58h]
  _BYTE v37[80]; // [rsp+40h] [rbp-50h] BYREF

  v3 = *(_QWORD *)(a1 + 32);
  v4 = v3 + 40LL * a2;
  if ( (*(_WORD *)(v4 + 2) & 0xFF0) != 0xFF0 )
    return (unsigned int)(unsigned __int8)(*(_WORD *)(v4 + 2) >> 4) - 1;
  v7 = *(unsigned __int16 *)(a1 + 68);
  if ( v7 != 1 )
  {
    if ( v7 == 32 )
    {
      v33 = a1;
      v34 = sub_2E88FE0(a1) + *(unsigned __int8 *)(*(_QWORD *)(a1 + 16) + 9LL);
      v32 = sub_2FC8970(&v33);
      v27 = sub_2E88FE0(a1) + *(unsigned __int8 *)(*(_QWORD *)(a1 + 16) + 9LL);
      if ( v27 )
      {
        v5 = v32;
        for ( i = 0; i != v27; ++i )
        {
          while ( *(_BYTE *)(*(_QWORD *)(a1 + 32) + 40LL * v5) )
            v5 = sub_2FC88B0(a1);
          if ( a2 == i )
            return v5;
          if ( a2 == v5 )
            return i;
          v5 = sub_2FC88B0(a1);
        }
      }
      goto LABEL_44;
    }
    if ( v7 != 2 )
    {
      if ( (*(_BYTE *)(v4 + 3) & 0x10) == 0 )
        return 254;
      v8 = *(_DWORD *)(a1 + 40) & 0xFFFFFF;
      if ( v8 != 254 )
      {
        v5 = 254;
        v9 = a2 + 1;
        do
        {
          v10 = v3 + 40LL * v5;
          if ( !*(_BYTE *)v10 && (*(_BYTE *)(v10 + 3) & 0x10) == 0 && (unsigned __int8)(*(_WORD *)(v10 + 2) >> 4) == v9 )
            return v5;
        }
        while ( v8 != ++v5 );
      }
LABEL_44:
      BUG();
    }
  }
  v11 = *(_DWORD *)(a1 + 40);
  v12 = v37;
  v35 = v37;
  v13 = v11 & 0xFFFFFF;
  v36 = 0x800000000LL;
  if ( (unsigned int)v13 <= 2 )
    goto LABEL_44;
  v14 = v3 + 80;
  v15 = 0;
  v16 = v37;
  v17 = 0xFFFFFFFFLL;
  v18 = 2;
  for ( j = 0; ; j = v21 )
  {
    *(_DWORD *)&v16[4 * v15] = v18;
    v21 = v36 + 1;
    v22 = *(_QWORD *)(v14 + 24);
    LODWORD(v36) = v36 + 1;
    v23 = ((unsigned __int16)v22 >> 3) + 1;
    if ( a2 <= v18 || v23 + v18 <= a2 )
      break;
    if ( (int)v22 < 0 )
    {
      v24 = (unsigned __int64)v35;
      v26 = *(_DWORD *)&v35[4 * (WORD1(v22) & 0x7FFF)];
LABEL_38:
      v5 = v26 + a2 - v18;
      goto LABEL_39;
    }
    v18 += v23;
    v17 = j;
LABEL_20:
    if ( (unsigned int)v13 <= v18 )
      goto LABEL_44;
    v14 = *(_QWORD *)(a1 + 32) + 40LL * v18;
    v15 = v21;
    v20 = v21 + 1LL;
    if ( v20 > HIDWORD(v36) )
    {
      v31 = v12;
      v29 = v17;
      v30 = v13;
      sub_C8D5F0((__int64)&v35, v12, v20, 4u, v17, v13);
      v15 = (unsigned int)v36;
      v17 = v29;
      v13 = v30;
      v12 = v31;
    }
    v16 = v35;
  }
  if ( (int)v22 >= 0 )
    goto LABEL_28;
  v24 = (unsigned __int64)v35;
  v25 = WORD1(v22) & 0x7FFF;
  v26 = *(_DWORD *)&v35[4 * v25];
  if ( j == (_DWORD)v17 )
    goto LABEL_38;
  if ( v25 != (_DWORD)v17 )
  {
LABEL_28:
    v18 += v23;
    goto LABEL_20;
  }
  v5 = a2 + v18 - v26;
LABEL_39:
  if ( (_BYTE *)v24 != v12 )
    _libc_free(v24);
  return v5;
}
