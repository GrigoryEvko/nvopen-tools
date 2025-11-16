// Function: sub_171A810
// Address: 0x171a810
//
__int64 __fastcall sub_171A810(__int64 a1, __int64 a2, __int64 a3, double a4, double a5, double a6)
{
  __int64 result; // rax
  unsigned __int8 v7; // r13
  _QWORD *v10; // rdi
  __int64 v11; // r14
  __int64 v12; // r15
  void *v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  char v19; // bl
  _QWORD *v20; // rdi
  __int64 v21; // r12
  __int64 v22; // r13
  void *v23; // rax
  int v24; // eax
  __int64 v25; // rdx
  __int64 v26; // rdi
  char v27; // [rsp-60h] [rbp-60h]
  __int64 v28; // [rsp-60h] [rbp-60h]
  __int64 v29; // [rsp-58h] [rbp-58h] BYREF
  _QWORD v30[10]; // [rsp-50h] [rbp-50h] BYREF

  if ( !a1 )
    return 0;
  result = 0;
  v7 = *(_BYTE *)(a1 + 16);
  if ( v7 > 0x17u )
  {
    if ( (v7 & 0xFD) == 0x24 )
    {
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v10 = *(_QWORD **)(a1 - 8);
      else
        v10 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
      v11 = *v10;
      v12 = v10[3];
      if ( *(_BYTE *)(*v10 + 16LL) == 14 )
      {
        v27 = *(_BYTE *)(v12 + 16);
        v13 = sub_16982C0();
        v14 = v11 + 32;
        if ( *(void **)(v11 + 32) == v13 )
          v14 = *(_QWORD *)(v11 + 40) + 8LL;
        if ( (*(_BYTE *)(v14 + 18) & 7) == 3 )
        {
          v15 = 0;
          if ( v27 != 14 )
          {
            a3 = a2;
            v18 = v12;
            v19 = 0;
            goto LABEL_36;
          }
        }
        else
        {
          v15 = v11;
          if ( v27 != 14 )
          {
            v17 = v12;
            v12 = 0;
LABEL_16:
            v28 = v17;
            sub_171A6E0(a2 + 8, v11 + 24);
            *(_QWORD *)a2 = 0;
            v18 = v28;
            goto LABEL_17;
          }
        }
      }
      else
      {
        if ( *(_BYTE *)(v12 + 16) != 14 )
        {
          v25 = 0;
          goto LABEL_44;
        }
        v23 = sub_16982C0();
        v15 = v11;
        v11 = 0;
        v13 = v23;
      }
      v16 = v12 + 32;
      if ( v13 == *(void **)(v12 + 32) )
        v16 = *(_QWORD *)(v12 + 40) + 8LL;
      if ( (*(_BYTE *)(v16 + 18) & 7) == 3 )
      {
        v17 = 0;
        if ( !v15 )
        {
          if ( v13 == *(void **)(v11 + 32) )
            sub_169C4E0(v30, (__int64)v13);
          else
            sub_1698360((__int64)v30, *(_QWORD *)(v11 + 32));
          sub_171A6E0(a2 + 8, (__int64)&v29);
          *(_QWORD *)a2 = 0;
          sub_127D120(v30);
          return 1;
        }
      }
      else
      {
        if ( !v15 )
        {
          a3 = a2;
          v19 = 0;
          goto LABEL_19;
        }
        v17 = v12;
      }
      if ( v11 )
        goto LABEL_16;
      v25 = v12;
      v11 = v15;
      v12 = v17;
LABEL_44:
      v18 = v12;
      *(_BYTE *)(a2 + 8) = 0;
      v12 = v25;
      *(_WORD *)(a2 + 10) = 1;
      *(_QWORD *)a2 = v11;
LABEL_17:
      if ( v18 )
      {
        v19 = 1;
        if ( v12 )
        {
LABEL_19:
          sub_171A6E0(a3 + 8, v12 + 24);
          *(_QWORD *)a3 = 0;
          if ( v7 != 38 )
            goto LABEL_20;
          if ( *(_BYTE *)(a3 + 8) )
          {
            v26 = a3 + 24;
            if ( *(void **)(a3 + 24) == sub_16982C0() )
              sub_169C8D0(v26, a4, a5, a6);
            else
              sub_1699490(v26);
LABEL_20:
            result = 2;
            if ( v19 )
              return result;
            return 1;
          }
          v24 = -*(unsigned __int16 *)(a3 + 10);
LABEL_38:
          *(_WORD *)(a3 + 10) = v24;
          goto LABEL_20;
        }
LABEL_36:
        *(_BYTE *)(a3 + 8) = 0;
        *(_WORD *)(a3 + 10) = 1;
        *(_QWORD *)a3 = v18;
        if ( v7 != 38 )
          goto LABEL_20;
        LOWORD(v24) = -1;
        goto LABEL_38;
      }
      return 1;
    }
    if ( v7 == 40 )
    {
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v20 = *(_QWORD **)(a1 - 8);
      else
        v20 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
      v21 = *v20;
      v22 = v20[3];
      if ( *(_BYTE *)(*v20 + 16LL) == 14 )
      {
        sub_171A6E0(a2 + 8, v21 + 24);
        *(_QWORD *)a2 = v22;
        return 1;
      }
      else
      {
        result = 0;
        if ( *(_BYTE *)(v22 + 16) == 14 )
        {
          sub_171A6E0(a2 + 8, v22 + 24);
          *(_QWORD *)a2 = v21;
          return 1;
        }
      }
    }
  }
  return result;
}
