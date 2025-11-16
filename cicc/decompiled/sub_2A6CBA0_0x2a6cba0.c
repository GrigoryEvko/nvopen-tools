// Function: sub_2A6CBA0
// Address: 0x2a6cba0
//
__int64 __fastcall sub_2A6CBA0(__int64 a1, unsigned __int8 *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rbx
  char v5; // al
  int v8; // eax
  unsigned __int64 v9; // rdx
  int v11; // ecx
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rsi
  int v16; // r13d
  unsigned int v17; // ebx
  unsigned __int8 *v18; // rax
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rdx
  __int64 v23; // r8
  unsigned int v24; // edi
  __int64 *v25; // rsi
  __int64 v26; // r10
  int v27; // esi
  int v28; // r11d

  v4 = *((_QWORD *)a2 + 1);
  v5 = *(_BYTE *)(v4 + 8);
  if ( v5 == 7 )
    return 0;
  if ( v5 == 15 )
  {
    v8 = *a2;
    v9 = (unsigned int)(v8 - 34);
    if ( (unsigned __int8)(v8 - 34) <= 0x33u )
    {
      v14 = 0x8000000000041LL;
      if ( !_bittest64(&v14, v9)
        || (v15 = *((_QWORD *)a2 - 4)) == 0
        || *(_BYTE *)v15
        || *((_QWORD *)a2 + 10) != *(_QWORD *)(v15 + 24) )
      {
LABEL_17:
        v16 = *(_DWORD *)(v4 + 12);
        if ( !v16 )
          return 0;
        v17 = 0;
        while ( 1 )
        {
          v18 = sub_2A6A1C0(a1, a2, v17);
          if ( !*v18 )
            break;
          if ( ++v17 == v16 )
            return 0;
        }
        sub_2A634B0(a1, v18, (__int64)a2, v19, v20, v21);
        return 1;
      }
      if ( (unsigned __int8)sub_B19060(a1 + 360, v15, v9, a4) )
        return 0;
      LOBYTE(v8) = *a2;
    }
    if ( (unsigned __int8)(v8 - 93) <= 1u )
      return 0;
    goto LABEL_17;
  }
  if ( *(_BYTE *)sub_2A68BC0(a1, a2) )
    return 0;
  v11 = *a2;
  if ( (unsigned __int8)(v11 - 34) <= 0x33u )
  {
    v12 = 0x8000000000041LL;
    if ( _bittest64(&v12, (unsigned int)(v11 - 34)) )
    {
      v13 = *((_QWORD *)a2 - 4);
      if ( v13 )
      {
        if ( !*(_BYTE *)v13 && *(_QWORD *)(v13 + 24) == *((_QWORD *)a2 + 10) )
        {
          v22 = *(unsigned int *)(a1 + 256);
          v23 = *(_QWORD *)(a1 + 240);
          if ( (_DWORD)v22 )
          {
            v24 = (v22 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
            v25 = (__int64 *)(v23 + 16LL * v24);
            v26 = *v25;
            if ( v13 == *v25 )
            {
LABEL_27:
              if ( v25 != (__int64 *)(v23 + 16 * v22) )
                return 0;
            }
            else
            {
              v27 = 1;
              while ( v26 != -4096 )
              {
                v28 = v27 + 1;
                v24 = (v22 - 1) & (v27 + v24);
                v25 = (__int64 *)(v23 + 16LL * v24);
                v26 = *v25;
                if ( v13 == *v25 )
                  goto LABEL_27;
                v27 = v28;
              }
            }
          }
        }
      }
    }
    if ( (_BYTE)v11 == 61 )
      return 0;
  }
  sub_2A6A450(a1, (__int64)a2);
  return 1;
}
