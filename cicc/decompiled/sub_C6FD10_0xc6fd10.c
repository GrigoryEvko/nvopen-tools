// Function: sub_C6FD10
// Address: 0xc6fd10
//
__int64 __fastcall sub_C6FD10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  int v6; // eax
  unsigned int v8; // r14d
  _QWORD *v9; // rax
  unsigned __int64 v12; // rax
  int v13; // r8d
  unsigned int v16; // eax
  unsigned int v17; // r8d
  unsigned int v18; // eax
  int v19; // eax
  int v22; // r14d
  bool v23; // zf
  __int64 v24; // r13
  unsigned int v26; // edx
  unsigned int v29; // eax
  int v31; // eax
  unsigned __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rdx
  unsigned __int64 v36; // rax
  __int64 v37; // rcx
  unsigned int v38; // eax
  __int64 v39; // rdx
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rsi
  unsigned __int64 v43; // rdx
  __int64 v44; // rax
  unsigned int v45; // eax
  unsigned int v46; // [rsp+Ch] [rbp-34h]
  int v47; // [rsp+Ch] [rbp-34h]
  unsigned int v48; // [rsp+Ch] [rbp-34h]

  if ( a5 )
  {
    v8 = *(_DWORD *)(a3 + 24);
    v9 = *(_QWORD **)(a3 + 16);
    if ( v8 > 0x40 )
      v9 = (_QWORD *)*v9;
    if ( ((unsigned __int8)v9 & 1) != 0 )
    {
      v12 = *(_QWORD *)(a2 + 16);
      if ( *(_DWORD *)(a2 + 24) > 0x40u )
        *(_QWORD *)v12 |= 1uLL;
      else
        *(_QWORD *)(a2 + 16) = v12 | 1;
      v8 = *(_DWORD *)(a3 + 24);
    }
    if ( *(_DWORD *)(a3 + 8) > 0x40u )
    {
      v13 = sub_C445E0(a3);
      v16 = *(_DWORD *)(a4 + 24);
      if ( v16 > 0x40 )
        goto LABEL_14;
    }
    else
    {
      v13 = 64;
      _RDX = ~*(_QWORD *)a3;
      __asm { tzcnt   rax, rdx }
      if ( *(_QWORD *)a3 != -1 )
        v13 = _RAX;
      v16 = *(_DWORD *)(a4 + 24);
      if ( v16 > 0x40 )
      {
LABEL_14:
        v17 = v13 - sub_C44590(a4 + 16);
        if ( v8 > 0x40 )
          goto LABEL_15;
        goto LABEL_30;
      }
    }
    _RCX = *(_QWORD *)(a4 + 16);
    v26 = 64;
    __asm { tzcnt   rsi, rcx }
    if ( _RCX )
      v26 = _RSI;
    if ( v16 > v26 )
      v16 = v26;
    v17 = v13 - v16;
    if ( v8 > 0x40 )
    {
LABEL_15:
      v46 = v17;
      v18 = sub_C44590(a3 + 16);
      v17 = v46;
      v8 = v18;
      if ( *(_DWORD *)(a4 + 8) <= 0x40u )
        goto LABEL_16;
      goto LABEL_35;
    }
LABEL_30:
    _RDX = *(_QWORD *)(a3 + 16);
    v29 = 64;
    __asm { tzcnt   rcx, rdx }
    if ( _RDX )
      v29 = _RCX;
    if ( v8 > v29 )
      v8 = v29;
    if ( *(_DWORD *)(a4 + 8) <= 0x40u )
    {
LABEL_16:
      v19 = 64;
      _RDX = ~*(_QWORD *)a4;
      __asm { tzcnt   rcx, rdx }
      if ( *(_QWORD *)a4 != -1 )
        v19 = _RCX;
      v22 = v8 - v19;
      v23 = v17 == 0;
      if ( (v17 & 0x80000000) != 0 )
      {
LABEL_19:
        if ( v22 >= 0 )
        {
LABEL_20:
          v24 = *(unsigned int *)(a2 + 8);
          if ( (unsigned int)v24 > 0x40 )
          {
LABEL_21:
            if ( !(unsigned __int8)sub_C446A0((__int64 *)a2, (__int64 *)(a2 + 16)) )
            {
LABEL_22:
              LODWORD(v24) = *(_DWORD *)(a2 + 8);
LABEL_23:
              *(_DWORD *)(a1 + 8) = v24;
              goto LABEL_3;
            }
            memset(*(void **)a2, -1, 8 * (((unsigned __int64)(unsigned int)v24 + 63) >> 6));
            v24 = *(unsigned int *)(a2 + 8);
            v35 = *(_QWORD *)a2;
LABEL_42:
            v36 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v24;
            if ( (_DWORD)v24 )
            {
              if ( (unsigned int)v24 > 0x40 )
              {
                v37 = (unsigned int)((unsigned __int64)(v24 + 63) >> 6) - 1;
                *(_QWORD *)(v35 + 8 * v37) &= v36;
LABEL_45:
                v38 = *(_DWORD *)(a2 + 24);
                if ( v38 <= 0x40 )
                {
                  *(_QWORD *)(a2 + 16) = 0;
                  LODWORD(v24) = *(_DWORD *)(a2 + 8);
                  goto LABEL_23;
                }
                memset(*(void **)(a2 + 16), 0, 8 * (((unsigned __int64)v38 + 63) >> 6));
                goto LABEL_22;
              }
            }
            else
            {
              v36 = 0;
            }
            *(_QWORD *)a2 = v35 & v36;
            goto LABEL_45;
          }
          v34 = *(_QWORD *)a2;
LABEL_40:
          if ( (*(_QWORD *)(a2 + 16) & v34) == 0 )
            goto LABEL_23;
          *(_QWORD *)a2 = -1;
          v35 = -1;
          goto LABEL_42;
        }
        v41 = *(unsigned int *)(a2 + 8);
        if ( (unsigned int)v41 > 0x40 )
        {
          memset(*(void **)a2, -1, 8 * (((unsigned __int64)(unsigned int)v41 + 63) >> 6));
          v41 = *(unsigned int *)(a2 + 8);
          v42 = *(_QWORD *)a2;
        }
        else
        {
          *(_QWORD *)a2 = -1;
          v42 = -1;
        }
        v43 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v41;
        if ( (_DWORD)v41 )
        {
          if ( (unsigned int)v41 > 0x40 )
          {
            v44 = (unsigned int)((unsigned __int64)(v41 + 63) >> 6) - 1;
            *(_QWORD *)(v42 + 8 * v44) &= v43;
LABEL_60:
            v45 = *(_DWORD *)(a2 + 24);
            if ( v45 <= 0x40 )
            {
              LODWORD(v24) = *(_DWORD *)(a2 + 8);
              *(_QWORD *)(a2 + 16) = 0;
              if ( (unsigned int)v24 <= 0x40 )
                goto LABEL_23;
              goto LABEL_21;
            }
            memset(*(void **)(a2 + 16), 0, 8 * (((unsigned __int64)v45 + 63) >> 6));
            goto LABEL_20;
          }
        }
        else
        {
          v43 = 0;
        }
        *(_QWORD *)a2 = v42 & v43;
        goto LABEL_60;
      }
LABEL_36:
      if ( !v23 )
      {
        if ( v17 > 0x40 )
        {
          v48 = v17;
          sub_C43C90((_QWORD *)a2, 0, v17);
          v17 = v48;
        }
        else
        {
          v24 = *(unsigned int *)(a2 + 8);
          v32 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v17);
          v33 = *(_QWORD *)a2;
          if ( (unsigned int)v24 <= 0x40 )
          {
            v34 = v32 | v33;
            *(_QWORD *)a2 = v34;
            if ( v17 != v22 )
              goto LABEL_40;
            goto LABEL_50;
          }
          *(_QWORD *)v33 |= v32;
        }
      }
      if ( v17 != v22 )
        goto LABEL_20;
LABEL_50:
      v39 = *(_QWORD *)(a2 + 16);
      v40 = 1LL << v17;
      if ( *(_DWORD *)(a2 + 24) > 0x40u )
        *(_QWORD *)(v39 + 8LL * (v17 >> 6)) |= v40;
      else
        *(_QWORD *)(a2 + 16) = v39 | v40;
      goto LABEL_20;
    }
LABEL_35:
    v47 = v17;
    v31 = sub_C445E0(a4);
    v17 = v47;
    v22 = v8 - v31;
    v23 = v47 == 0;
    if ( v47 < 0 )
      goto LABEL_19;
    goto LABEL_36;
  }
  *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
LABEL_3:
  *(_QWORD *)a1 = *(_QWORD *)a2;
  v6 = *(_DWORD *)(a2 + 24);
  *(_DWORD *)(a2 + 8) = 0;
  *(_DWORD *)(a1 + 24) = v6;
  *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
  *(_DWORD *)(a2 + 24) = 0;
  return a1;
}
