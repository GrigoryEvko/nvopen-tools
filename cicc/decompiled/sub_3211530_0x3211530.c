// Function: sub_3211530
// Address: 0x3211530
//
__int64 __fastcall sub_3211530(__int64 a1)
{
  __int64 result; // rax
  __int64 *v3; // rax
  __int64 v4; // rax
  _QWORD *v5; // rcx
  __int64 v6; // rdi
  __int64 v7; // rsi
  __int64 v8; // r9
  __int64 v9; // rdx
  _QWORD *v10; // r12
  _QWORD *v11; // r8
  __int64 v12; // r8
  __int64 v13; // rax
  unsigned __int8 v14; // dl
  __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rax
  int v21; // r10d

  result = *(_QWORD *)(a1 + 8);
  if ( !result || !*(_BYTE *)(result + 782) )
    return result;
  v3 = (__int64 *)sub_2E88D60(*(_QWORD *)(a1 + 64));
  v4 = sub_B92180(*v3);
  v5 = *(_QWORD **)(a1 + 64);
  v6 = v4;
  if ( (*(_BYTE *)(v5[2] + 24LL) & 0x10) == 0 )
  {
    *(_QWORD *)(a1 + 32) = 0;
    *(_QWORD *)(a1 + 40) = v5[3];
  }
  result = *(unsigned int *)(a1 + 488);
  v7 = *(_QWORD *)(a1 + 472);
  if ( !(_DWORD)result )
    goto LABEL_10;
  v8 = (unsigned int)(result - 1);
  v9 = (unsigned int)v8 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v10 = (_QWORD *)(v7 + 16 * v9);
  v11 = (_QWORD *)*v10;
  if ( v5 != (_QWORD *)*v10 )
  {
    v21 = 1;
    while ( v11 != (_QWORD *)-4096LL )
    {
      v9 = (unsigned int)v8 & (v21 + (_DWORD)v9);
      v10 = (_QWORD *)(v7 + 16LL * (unsigned int)v9);
      v11 = (_QWORD *)*v10;
      if ( v5 == (_QWORD *)*v10 )
        goto LABEL_8;
      ++v21;
    }
    goto LABEL_10;
  }
LABEL_8:
  result = v7 + 16 * result;
  if ( v10 == (_QWORD *)result || v10[1] )
  {
LABEL_10:
    *(_QWORD *)(a1 + 64) = 0;
    return result;
  }
  v12 = v5[3];
  if ( *(_BYTE *)(v12 + 261) && ((v13 = v5[1]) == 0 || (v9 = v12 + 48, v13 == v12 + 48)) )
  {
    result = sub_2E30F60(v5[3], v7, v9, (__int64)v5, v12, v8);
    *(_QWORD *)(a1 + 32) = result;
  }
  else
  {
    result = *(_QWORD *)(a1 + 32);
    if ( !result )
    {
      v14 = *(_BYTE *)(v6 - 16);
      v15 = (v14 & 2) != 0 ? *(_QWORD *)(v6 - 32) : v6 - 16 - 8LL * ((v14 >> 2) & 0xF);
      v16 = *(_QWORD *)(v15 + 40);
      if ( *(_DWORD *)(v16 + 32) != 3 )
      {
        v17 = *(_QWORD *)(a1 + 16);
        v18 = *(_QWORD *)(v17 + 2480);
        v19 = v17 + 8;
        if ( !v18 )
          v18 = v19;
        v20 = sub_E6C430(v18, v7, v16, (__int64)v5, v12);
        *(_QWORD *)(a1 + 32) = v20;
        (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 224LL) + 208LL))(
          *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL),
          v20,
          0);
        result = *(_QWORD *)(a1 + 32);
      }
    }
  }
  v10[1] = result;
  *(_QWORD *)(a1 + 64) = 0;
  return result;
}
