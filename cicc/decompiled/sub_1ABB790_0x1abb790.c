// Function: sub_1ABB790
// Address: 0x1abb790
//
__int64 __fastcall sub_1ABB790(__int64 a1, __int64 a2, _BYTE *a3, _BYTE *a4)
{
  __int64 v4; // r15
  __int64 v7; // r13
  __int64 v8; // r12
  _QWORD *v9; // rax
  unsigned __int8 v10; // cl
  int v12; // ecx
  __int64 v13; // rdi
  __int64 v14; // r8
  int v15; // ecx
  int v16; // r9d
  unsigned int v17; // eax
  __int64 v18; // rsi
  char v19; // al
  _BYTE *v20; // r11
  __int64 v21; // rcx
  int v22; // ecx

  v4 = *(_QWORD *)(a2 + 8);
  if ( v4 )
  {
    v7 = 0;
    v8 = 0;
    while ( 1 )
    {
      v9 = sub_1648700(v4);
      v10 = *((_BYTE *)v9 + 16);
      if ( v10 <= 0x17u )
        break;
      if ( v10 == 78 && (v21 = *(v9 - 3), !*(_BYTE *)(v21 + 16)) && (*(_BYTE *)(v21 + 33) & 0x20) != 0 )
      {
        v22 = *(_DWORD *)(v21 + 36);
        if ( v22 == 117 )
        {
          if ( v8 )
            return 0;
          v8 = (__int64)v9;
        }
        else if ( v22 == 116 )
        {
          if ( v7 )
            return 0;
          v7 = (__int64)v9;
        }
      }
      else
      {
        v12 = *(_DWORD *)(*(_QWORD *)a1 + 64LL);
        if ( !v12 )
          return 0;
        v13 = v9[5];
        v14 = *(_QWORD *)(*(_QWORD *)a1 + 48LL);
        v15 = v12 - 1;
        v16 = 1;
        v17 = v15 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v18 = *(_QWORD *)(v14 + 8LL * v17);
        if ( v18 != v13 )
        {
          while ( v18 != -8 )
          {
            v17 = v15 & (v16 + v17);
            v18 = *(_QWORD *)(v14 + 8LL * v17);
            if ( v13 == v18 )
              goto LABEL_8;
            ++v16;
          }
          return 0;
        }
      }
LABEL_8:
      v4 = *(_QWORD *)(v4 + 8);
      if ( !v4 )
      {
        if ( v8 && v7 )
        {
          *a3 = sub_1ABB580(*(_QWORD *)a1 + 40LL, v8) ^ 1;
          v19 = sub_1ABB580(*(_QWORD *)a1 + 40LL, v7) ^ 1;
          *a4 = v19;
          if ( !*v20 && !v19 )
            return v8;
          if ( (unsigned __int8)sub_1ABB600(*(_QWORD *)a1, a2) && (!*a4 || **(_QWORD **)(a1 + 8)) )
            return v8;
        }
        return 0;
      }
    }
  }
  return 0;
}
