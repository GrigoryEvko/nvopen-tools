// Function: sub_20DA600
// Address: 0x20da600
//
__int64 __fastcall sub_20DA600(int a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // rsi
  unsigned int v9; // r10d
  unsigned int v10; // r12d
  unsigned int v11; // edx
  __int16 v12; // r15
  _WORD *v13; // rdx
  unsigned __int16 *v14; // r11
  unsigned __int16 v15; // r15
  unsigned __int16 *v16; // r9
  unsigned __int16 *v17; // r14
  unsigned __int16 *v18; // rax
  __int64 v19; // rdx
  __int64 result; // rax
  __int64 v21; // rbx
  int v22; // eax
  unsigned __int16 *v23; // rax
  unsigned __int16 v24; // ax
  _QWORD *v25; // [rsp+10h] [rbp-50h]
  unsigned int v26; // [rsp+18h] [rbp-48h]
  unsigned int v27[4]; // [rsp+1Ch] [rbp-44h] BYREF
  unsigned int v28[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v27[0] = a1;
  if ( a1 <= 0 )
    return sub_1D041C0(a3, v27, a3, a4, a5);
  v6 = a2;
  if ( !a2 )
    BUG();
  v7 = *(_QWORD *)(a2 + 8);
  v8 = *(_QWORD *)(a2 + 56);
  v9 = 0;
  v10 = 0;
  v11 = *(_DWORD *)(v7 + 24LL * (unsigned int)a1 + 16);
  v12 = a1 * (v11 & 0xF);
  v13 = (_WORD *)(v8 + 2LL * (v11 >> 4));
  v14 = v13 + 1;
  v15 = *v13 + v12;
LABEL_4:
  v16 = v14;
  while ( 1 )
  {
    v17 = v16;
    if ( !v16 )
    {
      v19 = v9;
      result = 0;
      goto LABEL_8;
    }
    v18 = (unsigned __int16 *)(*(_QWORD *)(v6 + 48) + 4LL * v15);
    v19 = *v18;
    v10 = v18[1];
    if ( (_WORD)v19 )
      break;
LABEL_19:
    v24 = *v16;
    v14 = 0;
    ++v16;
    if ( !v24 )
      goto LABEL_4;
    v15 += v24;
  }
  while ( 1 )
  {
    result = v8 + 2LL * *(unsigned int *)(v7 + 24LL * (unsigned __int16)v19 + 8);
    if ( result )
      break;
    if ( !(_WORD)v10 )
    {
      v9 = v19;
      goto LABEL_19;
    }
    v19 = v10;
    v10 = 0;
  }
LABEL_8:
  while ( 1 )
  {
    v25 = (_QWORD *)v6;
    v21 = result;
    if ( !v17 )
      break;
    while ( 1 )
    {
      v26 = v19;
      v28[0] = (unsigned __int16)v19;
      v21 += 2;
      sub_1D041C0(a3, v28, v19, v6, v7);
      v22 = *(unsigned __int16 *)(v21 - 2);
      v19 = v26;
      v6 = (__int64)v25;
      if ( !(_WORD)v22 )
        break;
      v19 = v22 + v26;
    }
    if ( (_WORD)v10 )
    {
      result = v25[7] + 2LL * *(unsigned int *)(v25[1] + 24LL * (unsigned __int16)v10 + 8);
      v19 = v10;
      v10 = 0;
    }
    else
    {
      v10 = *v17;
      v15 += v10;
      if ( (_WORD)v10 )
      {
        ++v17;
        v23 = (unsigned __int16 *)(v25[6] + 4LL * v15);
        v19 = *v23;
        v10 = v23[1];
        result = v25[7] + 2LL * *(unsigned int *)(v25[1] + 24LL * (unsigned __int16)v19 + 8);
      }
      else
      {
        result = 0;
        v17 = 0;
      }
    }
  }
  return result;
}
