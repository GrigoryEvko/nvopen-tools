// Function: sub_625480
// Address: 0x625480
//
__int64 __fastcall sub_625480(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v4; // rbx
  unsigned int i; // edx
  __int64 result; // rax
  __int64 v7; // rdi
  __int64 v8; // r14
  __int64 v9; // r15
  __int64 *v10; // rsi
  __int64 *v11; // rdi
  __int64 v12; // r8
  unsigned int v13; // edx
  __int64 v14; // rcx
  __int64 j; // rbx
  _QWORD *v16; // rax
  int v17; // r13d
  __int64 v18; // r12
  unsigned int v19; // r10d
  unsigned __int64 v20; // rax
  unsigned int v21; // r10d
  unsigned int v22; // r11d
  unsigned __int64 *v23; // r9
  unsigned int v24; // eax
  unsigned __int64 *v25; // r9
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // [rsp+0h] [rbp-40h]
  unsigned int v30; // [rsp+Ch] [rbp-34h]

  v4 = a2 >> 3;
  for ( i = (a2 >> 3) & *(_DWORD *)(qword_4CFDE40 + 8); ; i = *(_DWORD *)(qword_4CFDE40 + 8) & (i + 1) )
  {
    result = *(_QWORD *)qword_4CFDE40 + 32LL * i;
    if ( a2 == *(_QWORD *)result )
      break;
    if ( !*(_QWORD *)result )
      return result;
  }
  v7 = *(_QWORD *)(result + 8);
  v8 = *(_QWORD *)(a2 + 8);
  if ( v7 && v8 )
  {
    v9 = *(_QWORD *)(a1 + 88);
    v29 = *(_QWORD *)(result + 16);
    v30 = *(_DWORD *)(result + 24);
    sub_866000(v7, 1, 0);
    sub_8600D0(1, v30, *(_QWORD *)(v9 + 152), 0);
    *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 11) |= 0x40u;
    if ( v29 )
      sub_886000(v29);
    *(_BYTE *)a2 &= ~0x20u;
    *(_QWORD *)(a2 + 8) = 0;
    if ( *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v9 + 152) + 168LL) + 56LL) == a2 )
      *(_BYTE *)(a1 + 104) &= ~2u;
    v10 = (__int64 *)v8;
    sub_625150(v9, v8, (_BYTE *)a2);
    v11 = (__int64 *)v8;
    sub_7AEB40(v8);
    v12 = qword_4CFDE40;
    v13 = *(_DWORD *)(qword_4CFDE40 + 8);
    v14 = *(_QWORD *)qword_4CFDE40;
    for ( j = v13 & (unsigned int)v4; ; j = v13 & ((_DWORD)j + 1) )
    {
      v16 = (_QWORD *)(v14 + 32LL * (unsigned int)j);
      if ( a2 == *v16 )
        break;
    }
    *v16 = 0;
    if ( *(_QWORD *)(v14 + 32LL * (((_DWORD)j + 1) & v13)) )
    {
      v17 = *(_DWORD *)(v12 + 8);
      v18 = *(_QWORD *)v12;
      v19 = v17 & (j + 1);
      v20 = *(_QWORD *)(*(_QWORD *)v12 + 32LL * v19);
      while ( 1 )
      {
        v24 = v17 & (v20 >> 3);
        LOBYTE(v10) = v19 < v24;
        v22 = v17 & (v19 + 1);
        v25 = (unsigned __int64 *)(v18 + 32LL * v22);
        if ( v24 <= (unsigned int)j && (unsigned __int8)v10 | (v19 > (unsigned int)j)
          || v19 > (unsigned int)j && v19 < v24 )
        {
          v11 = (__int64 *)(v18 + 32 * j);
          v10 = (__int64 *)(v18 + 32LL * v19);
          sub_622D80(v11, v10);
          v20 = *v23;
          if ( !*v23 )
            break;
          j = v21;
        }
        else
        {
          v20 = *v25;
          if ( !*v25 )
            break;
        }
        v19 = v22;
      }
    }
    --*(_DWORD *)(v12 + 12);
    sub_863FC0();
    return sub_866010(v11, v10, v26, v27, v28);
  }
  return result;
}
