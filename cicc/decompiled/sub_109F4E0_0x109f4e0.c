// Function: sub_109F4E0
// Address: 0x109f4e0
//
__int64 __fastcall sub_109F4E0(char *a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  char v4; // r13
  char *v7; // rdi
  _BYTE *v8; // r15
  char *v9; // r14
  void *v10; // rsi
  _BYTE *v11; // rax
  _BYTE *v12; // rdx
  void *v13; // rax
  char *v14; // rax
  char *v15; // rax
  char *v16; // rax
  char v17; // bl
  __int64 **v18; // rdi
  __int64 *v19; // r12
  __int64 *v20; // r13
  int v21; // eax
  char *v22; // rcx
  unsigned __int8 *v23; // rdi
  char v24; // [rsp-60h] [rbp-60h]
  char *v25; // [rsp-60h] [rbp-60h]
  __int64 v26[11]; // [rsp-58h] [rbp-58h] BYREF

  if ( !a1 )
    return 0;
  result = 0;
  v4 = *a1;
  if ( (unsigned __int8)*a1 > 0x1Cu )
  {
    if ( ((v4 - 43) & 0xFD) == 0 )
    {
      if ( (a1[7] & 0x40) != 0 )
        v7 = (char *)*((_QWORD *)a1 - 1);
      else
        v7 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
      v8 = *(_BYTE **)v7;
      v9 = (char *)*((_QWORD *)v7 + 4);
      if ( **(_BYTE **)v7 == 18 )
      {
        v24 = *v9;
        v10 = sub_C33340();
        v11 = v8 + 24;
        if ( *((void **)v8 + 3) == v10 )
          v11 = (_BYTE *)*((_QWORD *)v8 + 4);
        if ( (v11[20] & 7) == 3 )
        {
          if ( v24 != 18 )
          {
            a3 = a2;
            v16 = v9;
            v17 = 0;
            goto LABEL_37;
          }
          v12 = 0;
        }
        else
        {
          if ( v24 != 18 )
          {
            v15 = v9;
            v9 = 0;
LABEL_19:
            v25 = v15;
            sub_109E290(a2 + 8, (__int64 *)v8 + 3);
            *(_QWORD *)a2 = 0;
            v16 = v25;
            goto LABEL_20;
          }
          v12 = v8;
        }
      }
      else
      {
        if ( *v9 != 18 )
        {
          v22 = 0;
          goto LABEL_45;
        }
        v13 = sub_C33340();
        v12 = v8;
        v8 = 0;
        v10 = v13;
      }
      v14 = v9 + 24;
      if ( v10 == *((void **)v9 + 3) )
        v14 = (char *)*((_QWORD *)v9 + 4);
      if ( (v14[20] & 7) == 3 )
      {
        v15 = 0;
        if ( !v12 )
        {
          if ( *((void **)v8 + 3) == v10 )
            sub_C3C460(v26, (__int64)v10);
          else
            sub_C37380(v26, *((_QWORD *)v8 + 3));
          sub_109E290(a2 + 8, v26);
          *(_QWORD *)a2 = 0;
          sub_91D830(v26);
          return 1;
        }
      }
      else
      {
        if ( !v12 )
        {
          a3 = a2;
          v17 = 0;
          goto LABEL_22;
        }
        v15 = v9;
      }
      if ( v8 )
        goto LABEL_19;
      v22 = v9;
      v8 = v12;
      v9 = v15;
LABEL_45:
      v16 = v9;
      *(_BYTE *)(a2 + 8) = 0;
      v9 = v22;
      *(_WORD *)(a2 + 10) = 1;
      *(_QWORD *)a2 = v8;
LABEL_20:
      if ( v16 )
      {
        v17 = 1;
        if ( v9 )
        {
LABEL_22:
          sub_109E290(a3 + 8, (__int64 *)v9 + 3);
          *(_QWORD *)a3 = 0;
          if ( v4 != 45 )
            goto LABEL_23;
          if ( *(_BYTE *)(a3 + 8) )
          {
            v23 = (unsigned __int8 *)(a3 + 16);
            if ( *(void **)(a3 + 16) == sub_C33340() )
              sub_C3CCB0((__int64)v23);
            else
              sub_C34440(v23);
LABEL_23:
            result = 2;
            if ( v17 )
              return result;
            return 1;
          }
          v21 = -*(unsigned __int16 *)(a3 + 10);
LABEL_39:
          *(_WORD *)(a3 + 10) = v21;
          goto LABEL_23;
        }
LABEL_37:
        *(_BYTE *)(a3 + 8) = 0;
        *(_WORD *)(a3 + 10) = 1;
        *(_QWORD *)a3 = v16;
        if ( v4 != 45 )
          goto LABEL_23;
        LOWORD(v21) = -1;
        goto LABEL_39;
      }
      return 1;
    }
    if ( v4 == 47 )
    {
      if ( (a1[7] & 0x40) != 0 )
        v18 = (__int64 **)*((_QWORD *)a1 - 1);
      else
        v18 = (__int64 **)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
      v19 = *v18;
      v20 = v18[4];
      if ( *(_BYTE *)*v18 == 18 )
      {
        sub_109E290(a2 + 8, v19 + 3);
        *(_QWORD *)a2 = v20;
        return 1;
      }
      else
      {
        result = 0;
        if ( *(_BYTE *)v20 == 18 )
        {
          sub_109E290(a2 + 8, v20 + 3);
          *(_QWORD *)a2 = v19;
          return 1;
        }
      }
    }
  }
  return result;
}
