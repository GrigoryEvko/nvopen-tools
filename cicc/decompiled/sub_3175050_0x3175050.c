// Function: sub_3175050
// Address: 0x3175050
//
__int64 __fastcall sub_3175050(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r13d
  __int64 v4; // rbx
  __int64 v7; // rcx
  __int64 v8; // rdx
  unsigned int v9; // r12d
  __int64 v10; // rdx
  char v12; // al
  unsigned int v13; // esi
  __int64 v14; // r8
  unsigned int v15; // eax
  __int64 *v16; // rdi
  __int64 v17; // r10
  int v18; // edi
  int v19; // r11d
  __int64 v20; // [rsp+0h] [rbp-40h]
  __int64 v21; // [rsp+8h] [rbp-38h]

  v4 = *(_QWORD *)(a3 + 16);
  if ( !v4 )
    return 1;
  v7 = a3;
  while ( 1 )
  {
    v8 = *(_QWORD *)(v4 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v8 - 30) <= 0xAu )
      break;
    v4 = *(_QWORD *)(v4 + 8);
    if ( !v4 )
      return 1;
  }
  v9 = 0;
LABEL_7:
  if ( (unsigned int)qword_5034AC8 <= v9 )
    return 0;
  v10 = *(_QWORD *)(v8 + 40);
  LOBYTE(v3) = v7 == v10 || a2 == v10;
  if ( (_BYTE)v3 || (v20 = v7, v21 = v10, v12 = sub_2A64220(*(__int64 **)(a1 + 56), v10), v7 = v20, !v12) )
  {
LABEL_9:
    while ( 1 )
    {
      v4 = *(_QWORD *)(v4 + 8);
      if ( !v4 )
        return 1;
      v8 = *(_QWORD *)(v4 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v8 - 30) <= 0xAu )
      {
        ++v9;
        goto LABEL_7;
      }
    }
  }
  v13 = *(_DWORD *)(a1 + 120);
  v14 = *(_QWORD *)(a1 + 104);
  if ( v13 )
  {
    v15 = (v13 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
    v16 = (__int64 *)(v14 + 8LL * v15);
    v17 = *v16;
    if ( v21 == *v16 )
    {
LABEL_18:
      if ( v16 != (__int64 *)(v14 + 8LL * v13) )
        goto LABEL_9;
    }
    else
    {
      v18 = 1;
      while ( v17 != -4096 )
      {
        v19 = v18 + 1;
        v15 = (v13 - 1) & (v18 + v15);
        v16 = (__int64 *)(v14 + 8LL * v15);
        v17 = *v16;
        if ( v21 == *v16 )
          goto LABEL_18;
        v18 = v19;
      }
    }
  }
  return v3;
}
