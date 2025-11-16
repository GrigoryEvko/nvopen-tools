// Function: sub_5F90A0
// Address: 0x5f90a0
//
__int64 __fastcall sub_5F90A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  __int64 i; // rax
  __int64 v8; // r13
  unsigned __int8 *v9; // rbx
  __int64 v10; // r14
  _QWORD *v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 *v14; // r14
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // rdi

  result = dword_4D048B8;
  if ( dword_4D048B8 )
  {
    for ( i = *(_QWORD *)(a1 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v8 = *(_QWORD *)(i + 168);
    v9 = *(unsigned __int8 **)(v8 + 56);
    if ( !v9 )
      return (__int64)sub_5F8DB0(a1, 0);
    result = *v9;
    if ( (result & 8) == 0 )
    {
      if ( (result & 0x20) != 0 )
      {
        v11 = qword_4F04C68;
        v12 = qword_4F04C68[0] + 776LL * dword_4F04C64;
        if ( *(_BYTE *)(v12 + 4) != 6 )
          goto LABEL_26;
        while ( *(_BYTE *)(v12 - 772) == 6 )
          v12 -= 776;
        v13 = *(_QWORD *)a1;
        v14 = *(__int64 **)(*(_QWORD *)(**(_QWORD **)(v12 + 208) + 96LL) + 56LL);
        if ( !v14 )
          goto LABEL_26;
        while ( v13 != v14[2] || (v14[23] & 8) == 0 )
        {
          v14 = (__int64 *)*v14;
          if ( !v14 )
            goto LABEL_26;
        }
        v15 = *((_QWORD *)v9 + 1);
        if ( !v15 )
          goto LABEL_26;
        v16 = sub_877FE0(v13);
        sub_8600D0(1, *((unsigned int *)v14 + 16), v16, 0);
        v17 = v14[3];
        if ( v17 )
          sub_886000(v17);
        *v9 &= ~0x20u;
        a2 = v15;
        *((_QWORD *)v9 + 1) = 0;
        sub_625150(a1, v15, 0);
        sub_7AEB40(v15);
        *((_BYTE *)v14 + 184) &= ~8u;
        sub_863FC0();
        if ( (*v9 & 0x20) != 0 )
        {
LABEL_26:
          if ( (*(_BYTE *)(a1 + 195) & 3) == 1 )
            sub_894C00(*(_QWORD *)a1, a2, v11, a4, a5);
        }
      }
      *(_QWORD *)(v8 + 56) = 0;
      sub_5F8DB0(a1, 0);
      v10 = *(_QWORD *)(v8 + 56);
      if ( !(unsigned int)sub_8DAC40(v9, v10) )
      {
        result = sub_8DAC40(v10, v9);
        if ( !(_DWORD)result )
          goto LABEL_15;
      }
      result = dword_4F077BC;
      if ( dword_4F077BC )
      {
        if ( !dword_4F077B4 )
        {
          if ( qword_4F077A8 > 0x1869Fu )
          {
LABEL_15:
            *(_QWORD *)(v8 + 56) = v9;
            return result;
          }
          if ( (*(_BYTE *)(a1 + 195) & 3) == 1 )
          {
LABEL_14:
            *(_BYTE *)(a1 + 206) |= 0x10u;
            *(_BYTE *)(a1 + 193) |= 0x20u;
            goto LABEL_15;
          }
          if ( dword_4F077C4 != 2 )
          {
LABEL_13:
            if ( qword_4F077A8 > 0x9FC3u )
              goto LABEL_14;
LABEL_22:
            result = sub_6851C0(2367, a1 + 64);
            goto LABEL_15;
          }
LABEL_40:
          if ( unk_4F07778 > 202001 )
            goto LABEL_15;
          if ( unk_4F07778 > 201401 )
            goto LABEL_14;
          if ( !dword_4F077BC || dword_4F077B4 )
            goto LABEL_22;
          goto LABEL_13;
        }
      }
      else if ( !dword_4F077B4 )
      {
        goto LABEL_15;
      }
      if ( unk_4F077A0 > 0x138E3u )
        goto LABEL_15;
      if ( dword_4F077C4 != 2 )
        goto LABEL_22;
      goto LABEL_40;
    }
  }
  return result;
}
