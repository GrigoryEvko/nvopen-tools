// Function: sub_2C4FBF0
// Address: 0x2c4fbf0
//
__int64 __fastcall sub_2C4FBF0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  unsigned __int64 v4; // rax
  __int64 v5; // r8
  __int64 v6; // rdx
  __int64 v8; // rbx
  __int64 v9; // r15
  unsigned __int64 v10; // r11
  unsigned int v11; // r9d
  __int64 v12; // r13
  __int64 v13; // rdi
  _QWORD *v14; // r10
  unsigned __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rdx
  _QWORD *v18; // rdx
  __int64 v19; // rdi
  __int64 v21; // r9
  _QWORD *v22; // rdx
  unsigned __int64 v23; // [rsp+0h] [rbp-50h]
  _QWORD *v24; // [rsp+8h] [rbp-48h]
  const void *v25; // [rsp+10h] [rbp-40h]
  unsigned __int64 v26; // [rsp+18h] [rbp-38h]

  v5 = a2;
  v6 = 16 * a3;
  v25 = (const void *)(a1 + 16);
  *(_QWORD *)a1 = a1 + 16;
  v8 = a2 + v6;
  *(_QWORD *)(a1 + 8) = 0x300000000LL;
  if ( v5 != v5 + v6 )
  {
    v9 = a2;
    v10 = 3;
    v11 = 0;
    v12 = 32LL * a4;
    while ( 1 )
    {
      v14 = *(_QWORD **)v9;
      v16 = 0xFFFFFFFFLL;
      if ( *(_QWORD *)v9 )
      {
        v19 = *v14;
        if ( (*(_BYTE *)(*v14 + 7LL) & 0x40) != 0 )
          v13 = *(_QWORD *)(v19 - 8);
        else
          v13 = v19 - 32LL * (*(_DWORD *)(v19 + 4) & 0x7FFFFFF);
        v14 = sub_2C4FB60((_QWORD *)(v12 + v13), *(_DWORD *)(v9 + 8));
        v16 = (unsigned int)v15;
        v4 = v15;
      }
      v17 = v11;
      if ( v11 >= v10 )
      {
        v21 = v11 + 1LL;
        v4 = v16 | v26 & 0xFFFFFFFF00000000LL;
        v26 = v4;
        if ( v10 < v17 + 1 )
        {
          v23 = v4;
          v24 = v14;
          sub_C8D5F0(a1, v25, v17 + 1, 0x10u, v5, v21);
          v17 = *(unsigned int *)(a1 + 8);
          v4 = v23;
          v14 = v24;
        }
        v22 = (_QWORD *)(*(_QWORD *)a1 + 16 * v17);
        *v22 = v14;
        v22[1] = v4;
        ++*(_DWORD *)(a1 + 8);
      }
      else
      {
        v18 = (_QWORD *)(*(_QWORD *)a1 + 16LL * v11);
        if ( v18 )
        {
          v4 &= 0xFFFFFFFF00000000LL;
          *v18 = v14;
          v18[1] = v4 | v16;
          v11 = *(_DWORD *)(a1 + 8);
        }
        *(_DWORD *)(a1 + 8) = v11 + 1;
      }
      v9 += 16;
      if ( v8 == v9 )
        break;
      v11 = *(_DWORD *)(a1 + 8);
      v10 = *(unsigned int *)(a1 + 12);
    }
  }
  return a1;
}
