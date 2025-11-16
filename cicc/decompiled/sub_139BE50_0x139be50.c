// Function: sub_139BE50
// Address: 0x139be50
//
__int64 __fastcall sub_139BE50(_QWORD *a1, __int64 a2)
{
  __int64 v3; // r15
  __int64 v4; // rsi
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // r10
  unsigned int v9; // ecx
  __int64 *v10; // rdx
  __int64 v11; // rbx
  unsigned int v12; // r13d
  __int64 v14; // rax
  __int64 v15; // rax
  int v16; // r13d
  __int64 v17; // rdi
  __int64 v18; // rbx
  __int64 v19; // rax
  unsigned int v20; // r13d
  __int64 *v21; // r12
  __int64 v22; // rax
  __int64 v23; // rdx
  int v24; // edx
  int v25; // r13d
  int v26; // [rsp+1Ch] [rbp-144h]
  _BYTE *v27; // [rsp+20h] [rbp-140h] BYREF
  __int64 v28; // [rsp+28h] [rbp-138h]
  _BYTE v29[304]; // [rsp+30h] [rbp-130h] BYREF

  v3 = *(_QWORD *)(a2 + 40);
  v4 = a1[2];
  if ( v4 == a2 )
    return 0;
  v6 = a1[3];
  v7 = *(unsigned int *)(v6 + 48);
  if ( !(_DWORD)v7 )
    return 1;
  v8 = *(_QWORD *)(v6 + 32);
  v9 = (v7 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
  v10 = (__int64 *)(v8 + 16LL * v9);
  v11 = *v10;
  if ( v3 != *v10 )
  {
    v24 = 1;
    while ( v11 != -8 )
    {
      v25 = v24 + 1;
      v9 = (v7 - 1) & (v24 + v9);
      v10 = (__int64 *)(v8 + 16LL * v9);
      v11 = *v10;
      if ( v3 == *v10 )
        goto LABEL_4;
      v24 = v25;
    }
    return 1;
  }
LABEL_4:
  if ( v10 == (__int64 *)(v8 + 16 * v7) || !v10[1] )
    return 1;
  if ( v3 != *(_QWORD *)(v4 + 40) )
  {
    v12 = sub_15CCEE0(v6, v4, a2);
    if ( (_BYTE)v12 )
      return (unsigned int)sub_137E580(a2, a1[2], a1[3], 0) ^ 1;
    return v12;
  }
  if ( *(_BYTE *)(v4 + 16) == 29 )
    return 0;
  if ( *(_BYTE *)(a2 + 16) == 77 )
    return 0;
  if ( v4 == a2 )
    return 0;
  v12 = sub_143B490(a1[1], v4, a2);
  if ( !(_BYTE)v12 )
    return 0;
  v14 = *(_QWORD *)(*(_QWORD *)(v3 + 56) + 80LL);
  if ( !v14 || v3 != v14 - 24 )
  {
    v15 = sub_157EBA0(v3);
    if ( (unsigned int)sub_15F4D60(v15) )
    {
      v16 = 0;
      v27 = v29;
      v28 = 0x2000000000LL;
      v17 = sub_157EBA0(v3);
      if ( v17 )
      {
        v26 = sub_15F4D60(v17);
        v18 = sub_157EBA0(v3);
        v19 = (unsigned int)v28;
        if ( v26 > HIDWORD(v28) - (unsigned __int64)(unsigned int)v28 )
        {
          sub_16CD150(&v27, v29, v26 + (unsigned __int64)(unsigned int)v28, 8);
          v19 = (unsigned int)v28;
        }
        v20 = 0;
        v21 = (__int64 *)&v27[8 * v19];
        if ( v26 )
        {
          do
          {
            v22 = sub_15F4DF0(v18, v20);
            if ( v21 )
              *v21 = v22;
            ++v21;
            ++v20;
          }
          while ( v20 != v26 );
          v16 = v28 + v26;
        }
        else
        {
          v16 = v19;
        }
      }
      v23 = a1[3];
      LODWORD(v28) = v16;
      v12 = sub_137E120((__int64)&v27, v3, v23, 0) ^ 1;
      if ( v27 != v29 )
        _libc_free((unsigned __int64)v27);
    }
  }
  return v12;
}
