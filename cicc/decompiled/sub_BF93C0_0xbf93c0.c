// Function: sub_BF93C0
// Address: 0xbf93c0
//
void __fastcall sub_BF93C0(_BYTE *a1, __int64 a2)
{
  __int64 v3; // rdi
  unsigned int v4; // ebx
  __int64 v5; // r12
  unsigned int v6; // ebx
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // rsi
  __int64 *v10; // rax
  __int64 *v11; // rdx
  _BYTE *v12; // r13
  __int64 v13; // r12
  _BYTE *v14; // rax
  __int64 v15; // rax
  char v16; // dl
  __int64 v17; // r12
  _BYTE *v18; // rax
  __int64 v19; // rax
  const char *v20; // rax
  __int64 v21; // r12
  _BYTE *v22; // rax
  __int64 v23; // rax
  __int64 v25; // [rsp+8h] [rbp-188h]
  _QWORD v26[4]; // [rsp+10h] [rbp-180h] BYREF
  char v27; // [rsp+30h] [rbp-160h]
  char v28; // [rsp+31h] [rbp-15Fh]
  const char *v29; // [rsp+40h] [rbp-150h] BYREF
  __int64 *v30; // [rsp+48h] [rbp-148h]
  __int64 v31; // [rsp+50h] [rbp-140h]
  int v32; // [rsp+58h] [rbp-138h]
  char v33; // [rsp+5Ch] [rbp-134h]
  _BYTE v34[304]; // [rsp+60h] [rbp-130h] BYREF

  if ( *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL) == 7 )
  {
    v3 = *(_QWORD *)(a2 - 8);
    v4 = (*(_DWORD *)(a2 + 4) & 0x7FFFFFFu) >> 1;
    v5 = *(_QWORD *)(*(_QWORD *)v3 + 8LL);
    v29 = 0;
    v30 = (__int64 *)v34;
    v31 = 32;
    v32 = 0;
    v33 = 1;
    v6 = v4 - 1;
    if ( v6 )
    {
      v7 = v6;
      v8 = 1;
      v25 = v7;
      while ( 1 )
      {
        v9 = *(_QWORD *)(v3 + 32LL * (unsigned int)(2 * v8));
        if ( *(_BYTE *)v9 != 17 )
        {
          v28 = 1;
          v20 = "Case value is not a constant integer.";
          goto LABEL_35;
        }
        if ( v5 != *(_QWORD *)(v9 + 8) )
          break;
        if ( !v33 )
          goto LABEL_19;
        v10 = v30;
        v11 = &v30[HIDWORD(v31)];
        if ( v30 != v11 )
        {
          while ( v9 != *v10 )
          {
            if ( v11 == ++v10 )
              goto LABEL_22;
          }
LABEL_11:
          v12 = *(_BYTE **)(v3 + 32LL * (unsigned int)(2 * v8));
          v28 = 1;
          v26[0] = "Duplicate integer as switch case";
          v27 = 3;
          v13 = *(_QWORD *)a1;
          if ( *(_QWORD *)a1 )
          {
            v9 = *(_QWORD *)a1;
            sub_CA0E80(v26, *(_QWORD *)a1);
            v14 = *(_BYTE **)(v13 + 32);
            if ( (unsigned __int64)v14 >= *(_QWORD *)(v13 + 24) )
            {
              v9 = 10;
              sub_CB5D20(v13, 10);
            }
            else
            {
              *(_QWORD *)(v13 + 32) = v14 + 1;
              *v14 = 10;
            }
            v15 = *(_QWORD *)a1;
            a1[152] = 1;
            if ( v15 )
            {
              v9 = a2;
              sub_BDBD80((__int64)a1, (_BYTE *)a2);
              if ( v12 )
              {
                v9 = (__int64)v12;
                sub_BDBD80((__int64)a1, v12);
              }
            }
            goto LABEL_17;
          }
LABEL_33:
          a1[152] = 1;
          goto LABEL_17;
        }
LABEL_22:
        if ( HIDWORD(v31) < (unsigned int)v31 )
        {
          ++HIDWORD(v31);
          *v11 = v9;
          ++v29;
          if ( v25 == v8 )
            goto LABEL_24;
        }
        else
        {
LABEL_19:
          sub_C8CC70(&v29, v9);
          if ( !v16 )
          {
            v3 = *(_QWORD *)(a2 - 8);
            goto LABEL_11;
          }
          if ( v25 == v8 )
            goto LABEL_24;
        }
        v3 = *(_QWORD *)(a2 - 8);
        ++v8;
      }
      v28 = 1;
      v20 = "Switch constants must all be same type as switch value!";
LABEL_35:
      v26[0] = v20;
      v27 = 3;
      v21 = *(_QWORD *)a1;
      if ( !*(_QWORD *)a1 )
        goto LABEL_33;
      v9 = *(_QWORD *)a1;
      sub_CA0E80(v26, *(_QWORD *)a1);
      v22 = *(_BYTE **)(v21 + 32);
      if ( (unsigned __int64)v22 >= *(_QWORD *)(v21 + 24) )
      {
        v9 = 10;
        sub_CB5D20(v21, 10);
      }
      else
      {
        *(_QWORD *)(v21 + 32) = v22 + 1;
        *v22 = 10;
      }
      v23 = *(_QWORD *)a1;
      a1[152] = 1;
      if ( v23 )
      {
        v9 = a2;
        sub_BDBD80((__int64)a1, (_BYTE *)a2);
      }
LABEL_17:
      if ( !v33 )
        goto LABEL_18;
    }
    else
    {
LABEL_24:
      v9 = a2;
      sub_BF90E0(a1, a2);
      if ( !v33 )
LABEL_18:
        _libc_free(v30, v9);
    }
  }
  else
  {
    v17 = *(_QWORD *)a1;
    v34[1] = 1;
    v29 = "Switch must have void result type!";
    v34[0] = 3;
    if ( v17 )
    {
      sub_CA0E80(&v29, v17);
      v18 = *(_BYTE **)(v17 + 32);
      if ( (unsigned __int64)v18 >= *(_QWORD *)(v17 + 24) )
      {
        sub_CB5D20(v17, 10);
      }
      else
      {
        *(_QWORD *)(v17 + 32) = v18 + 1;
        *v18 = 10;
      }
      v19 = *(_QWORD *)a1;
      a1[152] = 1;
      if ( v19 )
        sub_BDBD80((__int64)a1, (_BYTE *)a2);
    }
    else
    {
      a1[152] = 1;
    }
  }
}
