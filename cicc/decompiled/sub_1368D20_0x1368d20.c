// Function: sub_1368D20
// Address: 0x1368d20
//
__int64 __fastcall sub_1368D20(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbp
  __int64 v4; // rdi
  int v5; // eax
  int v6; // ecx
  int v7; // r9d
  __int64 v8; // r8
  unsigned int v9; // ecx
  __int64 *v10; // rax
  __int64 v11; // r10
  int v13; // eax
  int v14; // r11d
  int v15; // [rsp-Ch] [rbp-Ch] BYREF
  __int64 v16; // [rsp-8h] [rbp-8h]

  v4 = *a1;
  if ( !v4 )
    return a2;
  v16 = v3;
  v5 = -1;
  v6 = *(_DWORD *)(v4 + 184);
  if ( v6 )
  {
    v7 = v6 - 1;
    v8 = *(_QWORD *)(v4 + 168);
    v9 = (v6 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v10 = (__int64 *)(v8 + 16LL * v9);
    v11 = *v10;
    if ( a3 == *v10 )
    {
LABEL_4:
      v5 = *((_DWORD *)v10 + 2);
    }
    else
    {
      v13 = 1;
      while ( v11 != -8 )
      {
        v14 = v13 + 1;
        v9 = v7 & (v13 + v9);
        v10 = (__int64 *)(v8 + 16LL * v9);
        v11 = *v10;
        if ( a3 == *v10 )
          goto LABEL_4;
        v13 = v14;
      }
      v5 = -1;
    }
  }
  v15 = v5;
  return sub_1370FE0(v4, a2, &v15);
}
