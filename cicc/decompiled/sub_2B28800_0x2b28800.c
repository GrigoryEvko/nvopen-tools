// Function: sub_2B28800
// Address: 0x2b28800
//
__int64 __fastcall sub_2B28800(__int64 a1, __int64 **a2)
{
  __int64 v3; // rax
  _BYTE *v4; // rdx
  __int64 v6; // rdi
  __int64 v7; // rcx
  __int64 *v8; // r8
  unsigned int v9; // esi
  __int64 *v10; // rbx
  __int64 v11; // r9
  __int64 v12; // r13
  _QWORD *v13; // rax
  __int64 v14; // rax
  char v15; // dl
  char v16; // al
  __int64 v17; // rdx
  __int64 v18; // rdx
  int v19; // r11d

  v3 = **a2;
  if ( !*(_DWORD *)(v3 + 104) )
  {
    v4 = *(_BYTE **)(v3 + 416);
    if ( *(_BYTE **)(v3 + 424) == v4 )
    {
      v6 = **(_QWORD **)v3;
      if ( *(_BYTE *)(*(_QWORD *)(v6 + 8) + 8LL) == 12 )
      {
        v7 = *((unsigned int *)a2 + 886);
        v8 = a2[441];
        if ( (_DWORD)v7 )
        {
          v9 = (v7 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
          v10 = &v8[3 * v9];
          v11 = *v10;
          if ( v3 == *v10 )
          {
LABEL_8:
            if ( v10 != &v8[3 * v7] )
            {
              v12 = v10[1];
              v13 = (_QWORD *)sub_BD5C60(v6);
              v14 = sub_BCCE00(v13, v12);
              v15 = *((_BYTE *)v10 + 16);
              *(_BYTE *)(a1 + 16) = 1;
              *(_QWORD *)a1 = v14;
              *(_BYTE *)(a1 + 8) = v15;
              return a1;
            }
          }
          else
          {
            v19 = 1;
            while ( v11 != -4096 )
            {
              v9 = (v7 - 1) & (v19 + v9);
              v10 = &v8[3 * v9];
              v11 = *v10;
              if ( v3 == *v10 )
                goto LABEL_8;
              ++v19;
            }
          }
        }
        v16 = *v4;
        if ( (unsigned __int8)(*v4 - 68) <= 1u )
        {
          v17 = *((_QWORD *)v4 - 4);
          *(_BYTE *)(a1 + 8) = v16 == 69;
          v18 = *(_QWORD *)(v17 + 8);
          *(_BYTE *)(a1 + 16) = 1;
          *(_QWORD *)a1 = v18;
          return a1;
        }
      }
    }
  }
  *(_BYTE *)(a1 + 16) = 0;
  return a1;
}
