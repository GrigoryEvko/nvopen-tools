// Function: sub_1473060
// Address: 0x1473060
//
__int64 __fastcall sub_1473060(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5, unsigned __int8 a6)
{
  __int64 *v10; // rax
  __int64 v11; // r15
  __int64 v12; // rsi
  unsigned int v13; // eax
  __int64 v14; // rdi
  unsigned int v15; // ecx
  __int64 v16; // rax
  __int64 v17; // r9
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // r9
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v29; // [rsp+10h] [rbp-90h] BYREF
  __int64 v30; // [rsp+18h] [rbp-88h]
  char v31; // [rsp+20h] [rbp-80h]
  _BYTE v32[8]; // [rsp+28h] [rbp-78h] BYREF
  __int64 v33; // [rsp+30h] [rbp-70h]
  unsigned __int64 v34; // [rsp+38h] [rbp-68h]

  if ( (*(_BYTE *)(a4 + 23) & 0x40) == 0 )
  {
    v10 = (__int64 *)(a4 - 24LL * (*(_DWORD *)(a4 + 20) & 0xFFFFFFF));
    if ( a5 != v10[3] )
      goto LABEL_3;
LABEL_20:
    v25 = sub_1456E90(a2);
    sub_14573F0(a1, v25);
    return a1;
  }
  v10 = *(__int64 **)(a4 - 8);
  if ( a5 == v10[3] )
    goto LABEL_20;
LABEL_3:
  v11 = sub_1472610(a2, *v10, a3);
  if ( (*(_BYTE *)(a4 + 23) & 0x40) != 0 )
    v12 = *(_QWORD *)(a4 - 8);
  else
    v12 = a4 - 24LL * (*(_DWORD *)(a4 + 20) & 0xFFFFFFF);
  if ( a5 != *(_QWORD *)(v12 + 24) )
  {
    v13 = (*(_DWORD *)(a4 + 20) & 0xFFFFFFFu) >> 1;
    v14 = v13 - 1;
    if ( v13 != 1 )
    {
      v15 = 2;
      v16 = 1;
      v17 = 0;
      while ( 1 )
      {
        while ( 1 )
        {
          v19 = 24;
          if ( (_DWORD)v16 != -1 )
            v19 = 24LL * (v15 + 1);
          if ( a5 == *(_QWORD *)(v12 + v19) )
            break;
          v18 = v16;
          v15 += 2;
          ++v16;
          if ( v14 == v18 )
            goto LABEL_14;
        }
        if ( v17 )
          break;
        v20 = v15;
        v21 = v16;
        v15 += 2;
        ++v16;
        v17 = *(_QWORD *)(v12 + 24 * v20);
        if ( v14 == v21 )
          goto LABEL_14;
      }
    }
  }
  v17 = 0;
LABEL_14:
  v22 = sub_145CE20(a2, v17);
  v23 = sub_14806B0(a2, v11, v22, 0, 0);
  sub_1472640((__int64)&v29, a2, v23, (_QWORD **)a3, a6, 0);
  if ( sub_14562D0(v29) && sub_14562D0(v30) )
  {
    v26 = sub_1456E90(a2);
    sub_14573F0(a1, v26);
  }
  else
  {
    *(_QWORD *)a1 = v29;
    *(_QWORD *)(a1 + 8) = v30;
    *(_BYTE *)(a1 + 16) = v31;
    sub_16CCEE0(a1 + 24, a1 + 64, 4, v32);
  }
  if ( v34 != v33 )
    _libc_free(v34);
  return a1;
}
