// Function: sub_384FBB0
// Address: 0x384fbb0
//
__int64 __fastcall sub_384FBB0(__int64 a1, __int64 a2)
{
  int v4; // edx
  unsigned int v5; // r12d
  __int64 v6; // rsi
  __int64 *v7; // rax
  char v8; // dl
  unsigned __int16 v9; // ax
  unsigned int v11; // eax
  unsigned int v12; // r12d
  __int64 *v13; // rdi
  unsigned int v14; // r8d
  __int64 *v15; // rcx
  __int64 v16; // [rsp+8h] [rbp-48h] BYREF
  _QWORD v17[2]; // [rsp+10h] [rbp-40h] BYREF
  __int64 v18; // [rsp+20h] [rbp-30h]
  __int64 v19; // [rsp+28h] [rbp-28h]

  v4 = *(_DWORD *)(a1 + 184);
  v17[0] = 0;
  v17[1] = -1;
  v18 = 0;
  v19 = 0;
  if ( !v4
    || !*(_DWORD *)(a1 + 216)
    || (LOBYTE(v11) = sub_384F1D0(a1, *(_QWORD *)(a2 - 24), &v16, v17), v12 = v11, !(_BYTE)v11) )
  {
    v5 = *(unsigned __int8 *)(a1 + 352);
    if ( !(_BYTE)v5 )
      return 0;
LABEL_4:
    v6 = *(_QWORD *)(a2 - 24);
    v7 = *(__int64 **)(a1 + 368);
    if ( *(__int64 **)(a1 + 376) == v7 )
    {
      v13 = &v7[*(unsigned int *)(a1 + 388)];
      v14 = *(_DWORD *)(a1 + 388);
      if ( v7 != v13 )
      {
        v15 = 0;
        while ( v6 != *v7 )
        {
          if ( *v7 == -2 )
            v15 = v7;
          if ( v13 == ++v7 )
          {
            if ( !v15 )
              goto LABEL_26;
            *v15 = v6;
            --*(_DWORD *)(a1 + 392);
            ++*(_QWORD *)(a1 + 360);
            return 0;
          }
        }
LABEL_6:
        v9 = *(_WORD *)(a2 + 18);
        if ( ((v9 >> 7) & 6) == 0 && (v9 & 1) == 0 )
        {
          *(_DWORD *)(a1 + 528) += 5;
          return v5;
        }
        return 0;
      }
LABEL_26:
      if ( v14 < *(_DWORD *)(a1 + 384) )
      {
        *(_DWORD *)(a1 + 388) = v14 + 1;
        *v13 = v6;
        ++*(_QWORD *)(a1 + 360);
        return 0;
      }
    }
    sub_16CCBA0(a1 + 360, v6);
    if ( v8 )
      return 0;
    goto LABEL_6;
  }
  if ( sub_15F32D0(a2) || (*(_BYTE *)(a2 + 18) & 1) != 0 )
  {
    sub_384F170(a1, v18);
    v5 = *(unsigned __int8 *)(a1 + 352);
    if ( !(_BYTE)v5 )
      return 0;
    goto LABEL_4;
  }
  *(_DWORD *)(v18 + 8) += 5;
  *(_DWORD *)(a1 + 556) += 5;
  return v12;
}
