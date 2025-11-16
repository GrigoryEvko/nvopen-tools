// Function: sub_1F34340
// Address: 0x1f34340
//
__int64 __fastcall sub_1F34340(__int64 a1, unsigned __int8 a2, __int64 *a3)
{
  unsigned int v5; // r13d
  __int64 v6; // rdi
  __int64 (*v7)(); // rax
  unsigned __int64 v8; // rax
  __int64 *v9; // rdi
  __int64 v10; // rdx
  __int16 v11; // ax
  __int64 v13; // r14
  unsigned int v14; // ebx
  __int16 v15; // ax
  unsigned __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int16 v19; // dx
  int v20; // eax
  __int16 v21; // dx
  __int64 v22; // r9
  __int64 i; // r8
  __int64 v24; // rcx
  __int64 v25; // rdi
  int v26; // r11d
  _DWORD *v27; // rsi
  __int64 v28; // rax
  __int16 v29; // ax
  __int16 v30; // ax
  __int64 v31; // rdx
  __int64 v32; // rax
  unsigned __int64 v33; // rsi
  _QWORD *v34; // rax
  _DWORD *v35; // rdi
  __int64 v36; // rcx
  __int64 v37; // rdx
  __int64 v38; // r13
  __int64 v39; // rax
  _DWORD *v40; // r8
  _DWORD *v41; // rdi
  __int64 v42; // rcx
  __int64 v43; // rdx
  unsigned __int8 v45; // [rsp+16h] [rbp-FAh]
  unsigned __int8 v46; // [rsp+17h] [rbp-F9h]
  __int64 *v47; // [rsp+18h] [rbp-F8h]
  __int64 v48; // [rsp+20h] [rbp-F0h] BYREF
  __int64 v49; // [rsp+28h] [rbp-E8h] BYREF
  unsigned __int64 v50[2]; // [rsp+30h] [rbp-E0h] BYREF
  _BYTE v51[208]; // [rsp+40h] [rbp-D0h] BYREF

  if ( !*(_BYTE *)(a1 + 49) && sub_1DD6C00(a3) )
    return 0;
  v46 = sub_1DD6970((__int64)a3, (__int64)a3);
  if ( v46 )
    return 0;
  v5 = *(_DWORD *)(a1 + 52);
  if ( !v5 )
  {
    v33 = sub_16D5D50();
    v34 = *(_QWORD **)&dword_4FA0208[2];
    if ( *(_QWORD *)&dword_4FA0208[2] )
    {
      v35 = dword_4FA0208;
      do
      {
        while ( 1 )
        {
          v36 = v34[2];
          v37 = v34[3];
          if ( v33 <= v34[4] )
            break;
          v34 = (_QWORD *)v34[3];
          if ( !v37 )
            goto LABEL_86;
        }
        v35 = v34;
        v34 = (_QWORD *)v34[2];
      }
      while ( v36 );
LABEL_86:
      if ( v35 != dword_4FA0208 && v33 >= *((_QWORD *)v35 + 4) )
      {
        v39 = *((_QWORD *)v35 + 7);
        v40 = v35 + 12;
        if ( v39 )
        {
          v41 = v35 + 12;
          do
          {
            while ( 1 )
            {
              v42 = *(_QWORD *)(v39 + 16);
              v43 = *(_QWORD *)(v39 + 24);
              if ( *(_DWORD *)(v39 + 32) >= dword_4FCB108 )
                break;
              v39 = *(_QWORD *)(v39 + 24);
              if ( !v43 )
                goto LABEL_106;
            }
            v41 = (_DWORD *)v39;
            v39 = *(_QWORD *)(v39 + 16);
          }
          while ( v42 );
LABEL_106:
          if ( v40 != v41 && dword_4FCB108 >= v41[8] && v41[9] )
            goto LABEL_109;
        }
      }
    }
    v38 = **(_QWORD **)(a1 + 40) + 112LL;
    if ( (unsigned __int8)sub_1560180(v38, 34) || (unsigned __int8)sub_1560180(v38, 17) )
    {
      v5 = 1;
    }
    else
    {
LABEL_109:
      v5 = *(_DWORD *)(a1 + 52);
      if ( !v5 )
        v5 = dword_4FCB1A0;
    }
  }
  v6 = *(_QWORD *)a1;
  v48 = 0;
  v50[0] = (unsigned __int64)v51;
  v49 = 0;
  v50[1] = 0x400000000LL;
  v7 = *(__int64 (**)())(*(_QWORD *)v6 + 264LL);
  if ( v7 != sub_1D820E0
    && !((unsigned __int8 (__fastcall *)(__int64, __int64 *, __int64 *, __int64 *, unsigned __int64 *, _QWORD))v7)(
          v6,
          a3,
          &v48,
          &v49,
          v50,
          0)
    || !sub_1DD6C00(a3) )
  {
    v47 = a3 + 3;
    v8 = a3[3] & 0xFFFFFFFFFFFFFFF8LL;
    v9 = (__int64 *)v8;
    if ( a3 + 3 == (__int64 *)v8 )
    {
      v45 = 0;
LABEL_18:
      v13 = a3[4];
      if ( v47 != (__int64 *)v13 )
      {
        v14 = 0;
        do
        {
          v15 = *(_WORD *)(v13 + 46);
          if ( (v15 & 4) != 0 || (v15 & 8) == 0 )
          {
            if ( (*(_QWORD *)(*(_QWORD *)(v13 + 16) + 8LL) & 0x80000LL) != 0 )
            {
LABEL_23:
              v16 = *(unsigned int *)(*(_QWORD *)(a3[7] + 8) + 516LL);
              if ( (unsigned int)v16 <= 0x1E )
              {
                v17 = 1610614920;
                if ( _bittest64(&v17, v16) )
                  goto LABEL_25;
              }
              v18 = *(_QWORD *)(v13 + 16);
              if ( *(_WORD *)v18 != 2 )
                goto LABEL_25;
LABEL_28:
              v19 = *(_WORD *)(v13 + 46);
              if ( (v19 & 4) != 0 )
                goto LABEL_63;
              goto LABEL_29;
            }
          }
          else if ( sub_1E15D00(v13, 0x80000u, 1) )
          {
            goto LABEL_23;
          }
          v18 = *(_QWORD *)(v13 + 16);
          if ( *(_WORD *)v18 != 1 )
            goto LABEL_28;
          if ( (*(_BYTE *)(*(_QWORD *)(v13 + 32) + 64LL) & 0x20) != 0 )
            goto LABEL_25;
          v19 = *(_WORD *)(v13 + 46);
          if ( (v19 & 4) != 0 )
          {
LABEL_63:
            v20 = *(_DWORD *)(v18 + 12) & 1;
            goto LABEL_31;
          }
LABEL_29:
          if ( (v19 & 8) == 0 )
            goto LABEL_63;
          LOBYTE(v20) = sub_1E15D00(v13, 0, 1);
LABEL_31:
          if ( (_BYTE)v20 || **(_WORD **)(v13 + 16) == 1 && sub_1E17880(v13) )
            goto LABEL_25;
          if ( *(_BYTE *)(a1 + 48) )
          {
            v29 = *(_WORD *)(v13 + 46);
            if ( (v29 & 4) != 0 )
            {
              v31 = *(_QWORD *)(v13 + 16);
              if ( (*(_BYTE *)(v31 + 8) & 8) != 0 )
                goto LABEL_25;
LABEL_71:
              v32 = (*(_QWORD *)(v31 + 8) >> 4) & 1LL;
            }
            else
            {
              if ( (v29 & 8) != 0 )
              {
                if ( sub_1E15D00(v13, 8u, 1) )
                  goto LABEL_25;
                if ( !*(_BYTE *)(a1 + 48) )
                  goto LABEL_34;
                v30 = *(_WORD *)(v13 + 46);
                if ( (v30 & 4) != 0 || (v30 & 8) == 0 )
                {
LABEL_70:
                  v31 = *(_QWORD *)(v13 + 16);
                  goto LABEL_71;
                }
              }
              else
              {
                if ( (*(_BYTE *)(*(_QWORD *)(v13 + 16) + 8LL) & 8) != 0 )
                  goto LABEL_25;
                if ( (v29 & 8) == 0 )
                  goto LABEL_70;
              }
              LOBYTE(v32) = sub_1E15D00(v13, 0x10u, 1);
            }
            if ( (_BYTE)v32 )
              goto LABEL_25;
          }
LABEL_34:
          v21 = **(_WORD **)(v13 + 16);
          if ( v21 != 45 && **(_WORD **)(v13 + 16) )
          {
            switch ( v21 )
            {
              case 2:
              case 3:
              case 4:
              case 6:
              case 9:
              case 12:
              case 13:
              case 17:
              case 18:
                break;
              default:
                ++v14;
                break;
            }
          }
          if ( v5 < v14 )
            goto LABEL_25;
          if ( (*(_BYTE *)v13 & 4) == 0 )
          {
            while ( (*(_BYTE *)(v13 + 46) & 8) != 0 )
              v13 = *(_QWORD *)(v13 + 8);
          }
          v13 = *(_QWORD *)(v13 + 8);
        }
        while ( v47 != (__int64 *)v13 );
      }
      v22 = a3[12];
      for ( i = a3[11]; v22 != i; i += 8 )
      {
        v24 = *(_QWORD *)(*(_QWORD *)i + 32LL);
        v25 = *(_QWORD *)i + 24LL;
        if ( v24 != v25 )
        {
          while ( **(_WORD **)(v24 + 16) == 45 || !**(_WORD **)(v24 + 16) )
          {
            v26 = *(_DWORD *)(v24 + 40);
            v27 = *(_DWORD **)(v24 + 32);
            if ( v26 != 1 )
            {
              v28 = 1;
              while ( a3 != *(__int64 **)&v27[10 * (unsigned int)(v28 + 1) + 6] )
              {
                v28 = (unsigned int)(v28 + 2);
                if ( v26 == (_DWORD)v28 )
                  goto LABEL_52;
              }
              v27 += 10 * v28;
            }
LABEL_52:
            if ( (*v27 & 0xFFF00) != 0 )
              goto LABEL_25;
            if ( (*(_BYTE *)v24 & 4) == 0 )
            {
              while ( (*(_BYTE *)(v24 + 46) & 8) != 0 )
                v24 = *(_QWORD *)(v24 + 8);
            }
            v24 = *(_QWORD *)(v24 + 8);
            if ( v24 == v25 )
              break;
          }
        }
      }
      v46 = v45;
      if ( !v45 )
      {
        v46 = a2;
        if ( !a2 )
        {
          if ( *(_BYTE *)(a1 + 48) )
            v46 = sub_1F34230((__int64 *)a1, (__int64)a3);
          else
            v46 = 1;
        }
      }
      goto LABEL_25;
    }
    if ( !v8 )
      BUG();
    v10 = *(_QWORD *)v8;
    v11 = *(_WORD *)(v8 + 46);
    if ( (v10 & 4) != 0 )
    {
      if ( (v11 & 4) != 0 )
        goto LABEL_11;
    }
    else if ( (v11 & 4) != 0 )
    {
      while ( 1 )
      {
        v9 = (__int64 *)(v10 & 0xFFFFFFFFFFFFFFF8LL);
        v11 = *(_WORD *)((v10 & 0xFFFFFFFFFFFFFFF8LL) + 46);
        if ( (v11 & 4) == 0 )
          break;
        v10 = *v9;
      }
    }
    if ( (v11 & 8) != 0 )
    {
      v45 = sub_1E15D00((__int64)v9, 0x100u, 1);
      goto LABEL_12;
    }
LABEL_11:
    v45 = BYTE1(*(_QWORD *)(v9[2] + 8)) & 1;
LABEL_12:
    if ( v45 && *(_BYTE *)(a1 + 48) )
    {
      v45 = *(_BYTE *)(a1 + 48);
      v5 = dword_4FCB0C0;
    }
    goto LABEL_18;
  }
LABEL_25:
  if ( (_BYTE *)v50[0] != v51 )
    _libc_free(v50[0]);
  return v46;
}
