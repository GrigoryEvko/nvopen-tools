// Function: sub_1056220
// Address: 0x1056220
//
bool __fastcall sub_1056220(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r10
  __int64 v4; // r9
  __int64 v5; // rax
  __int64 v6; // r8
  __int64 *i; // r11
  _QWORD *v8; // rdi
  _QWORD *v9; // rsi
  bool result; // al
  __int64 v11; // rax
  int v12; // edx
  __int64 v13; // rsi
  __int64 v14; // rcx
  __int64 *v15; // r12
  int v16; // ecx
  unsigned int v17; // edx
  __int64 *v18; // rdi
  __int64 v19; // rbx
  int v20; // edi
  int v21; // r13d
  __int64 v22; // [rsp+8h] [rbp-28h] BYREF

  v3 = a2;
  v4 = a3;
  v5 = 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v6 = *(_QWORD *)(a2 - 8);
    v3 = v6 + v5;
  }
  else
  {
    v6 = a2 - v5;
  }
  for ( i = &v22; v3 != v6; v6 += 32 )
  {
    if ( **(_BYTE **)v6 > 0x1Cu )
    {
      v11 = *(_QWORD *)(*(_QWORD *)v6 + 40LL);
      v12 = *(_DWORD *)(v4 + 72);
      v22 = v11;
      if ( v12 )
      {
        v13 = *(_QWORD *)(v4 + 64);
        v14 = *(unsigned int *)(v4 + 80);
        v15 = (__int64 *)(v13 + 8 * v14);
        if ( (_DWORD)v14 )
        {
          v16 = v14 - 1;
          v17 = v16 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v18 = (__int64 *)(v13 + 8LL * v17);
          v19 = *v18;
          if ( v11 == *v18 )
          {
LABEL_11:
            result = v15 != v18;
            if ( v15 != v18 )
              return result;
          }
          else
          {
            v20 = 1;
            while ( v19 != -4096 )
            {
              v21 = v20 + 1;
              v17 = v16 & (v20 + v17);
              v18 = (__int64 *)(v13 + 8LL * v17);
              v19 = *v18;
              if ( v11 == *v18 )
                goto LABEL_11;
              v20 = v21;
            }
          }
        }
      }
      else
      {
        v8 = *(_QWORD **)(v4 + 88);
        v9 = &v8[*(unsigned int *)(v4 + 96)];
        result = v9 != sub_1055FB0(v8, (__int64)v9, i);
        if ( result )
          return result;
      }
    }
  }
  return 0;
}
