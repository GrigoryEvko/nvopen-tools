// Function: sub_2D00530
// Address: 0x2d00530
//
__int64 __fastcall sub_2D00530(__int64 a1, __int64 a2, __int64 *a3, _DWORD *a4)
{
  char v8; // al
  unsigned int v9; // eax
  unsigned int v10; // r9d
  __int64 *v11; // r12
  __int64 v13; // rax
  int v14; // ecx
  __int64 v15; // r9
  signed __int64 v16; // r15
  void *v17; // rax
  void *v18; // rax
  unsigned __int64 v19; // r8
  unsigned __int64 v20; // rdi
  __int64 v21; // rdx
  _BYTE *v22; // rax
  unsigned __int8 v23; // al
  __int64 v24; // rsi
  unsigned __int8 v25; // al
  int v26; // ecx
  __int64 v27; // r8
  signed __int64 v28; // r13
  _QWORD *v29; // r15
  __int64 v30; // rax
  __int64 v31; // r14
  __int64 v32; // rax
  int v33; // eax
  __int64 v34; // r14
  __int64 v35; // rax
  __int64 v36; // [rsp+0h] [rbp-40h]
  unsigned __int8 v37; // [rsp+0h] [rbp-40h]
  int v38; // [rsp+0h] [rbp-40h]
  unsigned __int8 v39; // [rsp+0h] [rbp-40h]
  int v40; // [rsp+8h] [rbp-38h]
  _QWORD *v41; // [rsp+8h] [rbp-38h]
  unsigned __int8 v42; // [rsp+8h] [rbp-38h]

  while ( 1 )
  {
    while ( 1 )
    {
      v8 = *(_BYTE *)a2;
      if ( *(_BYTE *)a2 <= 0x1Cu )
        break;
      if ( (unsigned __int8)(v8 - 78) > 1u )
        goto LABEL_10;
      if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
        v11 = *(__int64 **)(a2 - 8);
      else
        v11 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
      a2 = *v11;
    }
    if ( v8 != 5 )
      goto LABEL_11;
    LOBYTE(v9) = sub_AC35E0(a2);
    v10 = v9;
    if ( !(_BYTE)v9 )
      break;
    a2 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    if ( *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL) != 14 )
      return 0;
  }
  if ( *(_WORD *)(a2 + 2) == 34 )
  {
    v24 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    if ( *(_BYTE *)(*(_QWORD *)(v24 + 8) + 8LL) == 14 )
    {
      v37 = v9;
      v25 = sub_2D00530(a1, v24, a3, a4);
      v10 = v37;
      v42 = v25;
      if ( v25 )
      {
        v26 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
        v27 = (unsigned int)(v26 - 1);
        v38 = v26;
        if ( v26 == 1 )
        {
          v34 = *(_QWORD *)(a1 + 248);
          v35 = sub_BB5290(a2);
          *a4 += sub_AE54E0(v34, v35, 0, 0);
        }
        else
        {
          v28 = 8 * v27;
          v29 = (_QWORD *)sub_22077B0(8 * v27);
          memset(v29, 0, v28);
          v30 = 1;
          do
          {
            v29[v30 - 1] = *(_QWORD *)(a2 + 32 * (v30 - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
            ++v30;
          }
          while ( (unsigned int)(v38 - 2) + 2LL != v30 );
          v31 = *(_QWORD *)(a1 + 248);
          v32 = sub_BB5290(a2);
          *a4 += sub_AE54E0(v31, v32, v29, v28 >> 3);
          j_j___libc_free_0((unsigned __int64)v29);
        }
        return v42;
      }
    }
  }
  else
  {
LABEL_10:
    if ( *(_BYTE *)a2 != 63 )
    {
LABEL_11:
      *a3 = a2;
      v10 = 1;
      *a4 = 0;
      return v10;
    }
    v13 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
    v14 = v13 - 1;
    v15 = *(_QWORD *)(a2 - 32 * v13);
    if ( (_DWORD)v13 == 1 )
    {
      v16 = 0;
      v19 = 0;
    }
    else
    {
      v40 = v13 - 1;
      v16 = 8LL * (unsigned int)(v13 - 1);
      v36 = *(_QWORD *)(a2 - 32 * v13);
      v17 = (void *)sub_22077B0(v16);
      v18 = memset(v17, 0, v16);
      v14 = v40;
      v15 = v36;
      v19 = (unsigned __int64)v18;
    }
    v20 = v19;
    LODWORD(v21) = 0;
    while ( (_DWORD)v21 != v14 )
    {
      v20 += 8LL;
      v21 = (unsigned int)(v21 + 1);
      v22 = *(_BYTE **)(a2 + 32 * (v21 - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
      *(_QWORD *)(v20 - 8) = v22;
      if ( *v22 != 17 )
        goto LABEL_20;
    }
    v41 = (_QWORD *)v19;
    v23 = sub_2D00530(a1, v15, a3, a4);
    v19 = (unsigned __int64)v41;
    if ( !v23 )
    {
LABEL_20:
      v10 = 0;
      goto LABEL_21;
    }
    v39 = v23;
    v33 = sub_AE54E0(*(_QWORD *)(a1 + 248), *(_QWORD *)(a2 + 72), v41, v16 >> 3);
    v10 = v39;
    *a4 += v33;
    v19 = (unsigned __int64)v41;
LABEL_21:
    if ( v19 )
    {
      v42 = v10;
      j_j___libc_free_0(v19);
      return v42;
    }
  }
  return v10;
}
