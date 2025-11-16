// Function: sub_9B8B30
// Address: 0x9b8b30
//
__int64 __fastcall sub_9B8B30(__int64 a1, __int64 a2)
{
  char v4; // r14
  char v5; // bl
  char v6; // r15
  __int64 v7; // r14
  __int64 v8; // rsi
  __int64 v9; // rax
  unsigned __int8 v10; // cl
  unsigned __int64 v11; // rdx
  __int64 *v12; // r15
  __int64 *v13; // rbx
  char v14; // di
  __int64 *v15; // rax
  __int64 v16; // rcx
  unsigned int v17; // eax
  __int64 *v18; // rax
  unsigned __int64 v19; // rdx
  __int64 *v20; // rax
  __int64 v21; // rax
  __int64 *v23; // r15
  __int64 *v24; // r13
  __int64 v25; // rbx
  __int64 *v26; // rax
  __int64 v27; // rax
  unsigned __int64 v28; // rdx
  __int64 *v29; // rdi
  __int64 v30; // rax
  __int64 v31; // rax
  bool v32; // zf
  __int64 *v33; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v34; // [rsp+18h] [rbp-98h]
  _BYTE v35[32]; // [rsp+20h] [rbp-90h] BYREF
  __int64 v36; // [rsp+40h] [rbp-70h] BYREF
  __int64 *v37; // [rsp+48h] [rbp-68h]
  __int64 v38; // [rsp+50h] [rbp-60h]
  int v39; // [rsp+58h] [rbp-58h]
  char v40; // [rsp+5Ch] [rbp-54h]
  __int64 v41; // [rsp+60h] [rbp-50h] BYREF

  v4 = ((__int64 (*)(void))sub_B46420)();
  v5 = sub_B46420(a2);
  if ( !v4 && !(unsigned __int8)sub_B46490(a1) )
  {
    if ( (v5 || (unsigned __int8)sub_B46490(a2)) && (*(_BYTE *)(a2 + 7) & 0x20) != 0 )
      return sub_B91C10(a2, 25);
    return 0;
  }
  v6 = *(_BYTE *)(a1 + 7) & 0x20;
  if ( !v5 && !(unsigned __int8)sub_B46490(a2) )
  {
    if ( v6 )
      return sub_B91C10(a1, 25);
    return 0;
  }
  v7 = 0;
  if ( v6 )
    v7 = sub_B91C10(a1, 25);
  if ( (*(_BYTE *)(a2 + 7) & 0x20) == 0 )
    return 0;
  v8 = 25;
  v9 = sub_B91C10(a2, 25);
  if ( !v7 || !v9 )
    return 0;
  if ( v9 != v7 )
  {
    v36 = 0;
    v37 = &v41;
    v38 = 4;
    v39 = 0;
    v40 = 1;
    v10 = *(_BYTE *)(v9 - 16);
    if ( (v10 & 2) != 0 )
    {
      v11 = *(unsigned int *)(v9 - 24);
      if ( (_DWORD)v11 )
      {
        v12 = *(__int64 **)(v9 - 32);
LABEL_14:
        v13 = &v12[v11];
        v14 = 1;
        while ( 1 )
        {
          v8 = *v12;
          if ( !v14 )
            goto LABEL_38;
          v15 = v37;
          v11 = (unsigned __int64)&v37[HIDWORD(v38)];
          if ( v37 != (__int64 *)v11 )
          {
            while ( v8 != *v15 )
            {
              if ( (__int64 *)v11 == ++v15 )
                goto LABEL_39;
            }
            goto LABEL_20;
          }
LABEL_39:
          if ( HIDWORD(v38) < (unsigned int)v38 )
          {
            ++HIDWORD(v38);
            *(_QWORD *)v11 = v8;
            v14 = v40;
            ++v36;
          }
          else
          {
LABEL_38:
            sub_C8CC70(&v36, v8);
            v14 = v40;
          }
LABEL_20:
          if ( v13 == ++v12 )
            goto LABEL_21;
        }
      }
    }
    else
    {
      v11 = (*(_WORD *)(v9 - 16) >> 6) & 0xF;
      if ( ((*(_WORD *)(v9 - 16) >> 6) & 0xF) != 0 )
      {
        v11 = (*(_WORD *)(v9 - 16) >> 6) & 0xF;
        v12 = (__int64 *)(v9 - 8LL * ((v10 >> 2) & 0xF) - 16);
        goto LABEL_14;
      }
    }
    HIDWORD(v38) = 1;
    v41 = v9;
    v36 = 1;
LABEL_21:
    v33 = (__int64 *)v35;
    v34 = 0x400000000LL;
    v16 = *(unsigned __int8 *)(v7 - 16);
    if ( (v16 & 2) != 0 )
    {
      v17 = *(_DWORD *)(v7 - 24);
      if ( !v17 )
        goto LABEL_23;
      v23 = *(__int64 **)(v7 - 32);
      v19 = v17;
    }
    else
    {
      if ( ((*(_WORD *)(v7 - 16) >> 6) & 0xF) == 0 )
      {
LABEL_23:
        if ( v40 )
        {
          v18 = v37;
          v19 = (unsigned __int64)&v37[HIDWORD(v38)];
          if ( v37 == (__int64 *)v19 )
          {
LABEL_73:
            v7 = 0;
LABEL_61:
            if ( !v40 )
              _libc_free(v37, v8);
            return v7;
          }
          while ( v7 != *v18 )
          {
            if ( (__int64 *)v19 == ++v18 )
              goto LABEL_73;
          }
          v20 = (__int64 *)v35;
        }
        else
        {
          v8 = v7;
          v32 = sub_C8CA60(&v36, v7, v11, v16) == 0;
          v21 = (unsigned int)v34;
          if ( v32 )
          {
LABEL_56:
            if ( v21 )
            {
              if ( v21 == 1 )
              {
                v29 = v33;
                v7 = *v33;
              }
              else
              {
                v30 = sub_BD5C60(a1, v8, v19);
                v8 = (__int64)v33;
                v31 = sub_B9C770(v30, v33, (unsigned int)v34, 0, 1);
                v29 = v33;
                v7 = v31;
              }
            }
            else
            {
              v29 = v33;
              v7 = 0;
            }
            if ( v29 != (__int64 *)v35 )
              _libc_free(v29, v8);
            goto LABEL_61;
          }
          if ( (unsigned __int64)(unsigned int)v34 + 1 <= HIDWORD(v34) )
          {
            v19 = (unsigned __int64)v33;
          }
          else
          {
            v8 = (__int64)v35;
            sub_C8D5F0(&v33, v35, (unsigned int)v34 + 1LL, 8);
            v19 = (unsigned int)v34;
          }
          v20 = &v33[(unsigned int)v34];
        }
        *v20 = v7;
        v21 = (unsigned int)(v34 + 1);
        LODWORD(v34) = v34 + 1;
        goto LABEL_56;
      }
      v19 = (*(_WORD *)(v7 - 16) >> 6) & 0xF;
      v16 = 8LL * (((unsigned __int8)v16 >> 2) & 0xF);
      v23 = (__int64 *)(v7 - v16 - 16);
    }
    v24 = &v23[v19];
    while ( 1 )
    {
      v25 = *v23;
      if ( v40 )
      {
        v26 = v37;
        v19 = (unsigned __int64)&v37[HIDWORD(v38)];
        if ( v37 != (__int64 *)v19 )
        {
          while ( v25 != *v26 )
          {
            if ( (__int64 *)v19 == ++v26 )
              goto LABEL_54;
          }
LABEL_51:
          v27 = (unsigned int)v34;
          v16 = HIDWORD(v34);
          v28 = (unsigned int)v34 + 1LL;
          if ( v28 > HIDWORD(v34) )
          {
            v8 = (__int64)v35;
            sub_C8D5F0(&v33, v35, v28, 8);
            v27 = (unsigned int)v34;
          }
          v19 = (unsigned __int64)v33;
          v33[v27] = v25;
          LODWORD(v34) = v34 + 1;
        }
      }
      else
      {
        v8 = *v23;
        if ( sub_C8CA60(&v36, v25, v19, v16) )
          goto LABEL_51;
      }
LABEL_54:
      if ( v24 == ++v23 )
      {
        v21 = (unsigned int)v34;
        goto LABEL_56;
      }
    }
  }
  return v7;
}
