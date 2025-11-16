// Function: sub_1368C40
// Address: 0x1368c40
//
__int64 __fastcall sub_1368C40(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v4; // r13
  __int64 v6; // rax
  int v7; // ecx
  __int64 v8; // rdx
  int v9; // eax
  int v10; // esi
  __int64 v11; // rdi
  unsigned int v12; // ecx
  __int64 *v13; // rax
  __int64 v14; // r8
  int v16; // eax
  int v17; // r9d
  int v18[9]; // [rsp+Ch] [rbp-24h] BYREF

  v4 = *a2;
  if ( *a2 )
  {
    v6 = sub_1368BD0(a2);
    v7 = *(_DWORD *)(v4 + 184);
    v8 = v6;
    v9 = -1;
    if ( v7 )
    {
      v10 = v7 - 1;
      v11 = *(_QWORD *)(v4 + 168);
      v12 = (v7 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v13 = (__int64 *)(v11 + 16LL * v12);
      v14 = *v13;
      if ( a3 == *v13 )
      {
LABEL_4:
        v9 = *((_DWORD *)v13 + 2);
      }
      else
      {
        v16 = 1;
        while ( v14 != -8 )
        {
          v17 = v16 + 1;
          v12 = v10 & (v16 + v12);
          v13 = (__int64 *)(v11 + 16LL * v12);
          v14 = *v13;
          if ( a3 == *v13 )
            goto LABEL_4;
          v16 = v17;
        }
        v9 = -1;
      }
    }
    v18[0] = v9;
    sub_1370E50(a1, v4, v8, v18);
    return a1;
  }
  else
  {
    *(_BYTE *)(a1 + 8) = 0;
    return a1;
  }
}
