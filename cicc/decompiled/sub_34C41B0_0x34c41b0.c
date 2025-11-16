// Function: sub_34C41B0
// Address: 0x34c41b0
//
__int64 __fastcall sub_34C41B0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, _DWORD *a5)
{
  __int64 *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r15
  __int64 *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r14
  __int64 v11; // r12
  unsigned int v12; // r13d
  __int16 v13; // ax
  __int64 *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 v17; // rax
  unsigned int v18; // r8d
  int v20; // eax
  int v21; // eax
  int v22; // eax
  __int64 v23; // rax
  __int64 v26; // [rsp+18h] [rbp-48h]
  unsigned int v27; // [rsp+28h] [rbp-38h]

  *a5 = 0;
  v5 = *(__int64 **)(a1 + 104);
  v6 = (__int64)(*(_QWORD *)(a1 + 112) - (_QWORD)v5) >> 4;
  if ( (_DWORD)v6 )
  {
    v27 = -1;
    v7 = 0;
    v26 = (unsigned int)v6;
    while ( 1 )
    {
      v8 = &v5[2 * v7];
      v9 = *(_QWORD *)(*v8 + 8);
      if ( *a2 == v9 )
      {
        *a5 = v7;
        v5 = (__int64 *)(16 * v7 + *(_QWORD *)(a1 + 104));
        goto LABEL_14;
      }
      v10 = v8[1];
      v11 = *(_QWORD *)(v9 + 56);
      v12 = 0;
      if ( v11 != v10 )
        break;
LABEL_42:
      v27 = v12;
      *a5 = v7;
LABEL_12:
      v5 = *(__int64 **)(a1 + 104);
      if ( v26 == ++v7 )
      {
        v5 += 2 * (unsigned int)*a5;
        goto LABEL_14;
      }
    }
    while ( 1 )
    {
      v13 = *(_WORD *)(v11 + 68);
      if ( (unsigned __int16)(v13 - 14) <= 4u || v13 == 3 )
        goto LABEL_9;
      v20 = *(_DWORD *)(v11 + 44);
      if ( (v20 & 4) != 0 || (v20 & 8) == 0 )
      {
        if ( (*(_QWORD *)(*(_QWORD *)(v11 + 16) + 24LL) & 0x80u) == 0LL )
        {
LABEL_28:
          if ( (unsigned int)*(unsigned __int16 *)(v11 + 68) - 1 <= 1
            && (*(_BYTE *)(*(_QWORD *)(v11 + 32) + 64LL) & 8) != 0 )
          {
            goto LABEL_33;
          }
          v21 = *(_DWORD *)(v11 + 44);
          if ( (v21 & 4) != 0 || (v21 & 8) == 0 )
          {
            if ( (*(_QWORD *)(*(_QWORD *)(v11 + 16) + 24LL) & 0x80000LL) != 0 )
              goto LABEL_33;
          }
          else if ( sub_2E88A90(v11, 0x80000, 1) )
          {
LABEL_33:
            v12 += 2;
            goto LABEL_9;
          }
          if ( (unsigned int)*(unsigned __int16 *)(v11 + 68) - 1 > 1
            || (*(_BYTE *)(*(_QWORD *)(v11 + 32) + 64LL) & 0x10) == 0 )
          {
            v22 = *(_DWORD *)(v11 + 44);
            if ( (v22 & 4) != 0 || (v22 & 8) == 0 )
              v23 = (*(_QWORD *)(*(_QWORD *)(v11 + 16) + 24LL) >> 20) & 1LL;
            else
              LOBYTE(v23) = sub_2E88A90(v11, 0x100000, 1);
            if ( !(_BYTE)v23 )
            {
              ++v12;
              goto LABEL_9;
            }
          }
          goto LABEL_33;
        }
      }
      else if ( !sub_2E88A90(v11, 128, 1) )
      {
        goto LABEL_28;
      }
      v12 += 10;
LABEL_9:
      if ( (*(_BYTE *)v11 & 4) != 0 )
      {
        v11 = *(_QWORD *)(v11 + 8);
        if ( v10 == v11 )
          goto LABEL_11;
      }
      else
      {
        while ( (*(_BYTE *)(v11 + 44) & 8) != 0 )
          v11 = *(_QWORD *)(v11 + 8);
        v11 = *(_QWORD *)(v11 + 8);
        if ( v10 == v11 )
        {
LABEL_11:
          if ( v12 > v27 )
            goto LABEL_12;
          goto LABEL_42;
        }
      }
    }
  }
LABEL_14:
  v14 = (__int64 *)v5[1];
  v15 = *v5;
  v16 = *(_QWORD *)(v15 + 8);
  if ( !a3 || *(_DWORD *)(v16 + 120) != 1 )
  {
    v17 = sub_34C3D00(a1, *(_QWORD **)(v15 + 8), v14, *(_QWORD *)(v16 + 16));
    if ( v17 )
      goto LABEL_17;
    return 0;
  }
  v17 = sub_34C3D00(a1, (_QWORD *)v16, v14, *(_QWORD *)(a3 + 16));
  if ( !v17 )
    return 0;
LABEL_17:
  v18 = 1;
  *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 104) + 16LL * (unsigned int)*a5) + 8LL) = v17;
  *(_QWORD *)(*(_QWORD *)(a1 + 104) + 16LL * (unsigned int)*a5 + 8) = *(_QWORD *)(v17 + 56);
  if ( *a2 == v16 )
    *a2 = v17;
  return v18;
}
