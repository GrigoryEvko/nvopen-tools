// Function: sub_164B430
// Address: 0x164b430
//
void __fastcall sub_164B430(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  unsigned __int8 v3; // al
  bool v4; // zf
  size_t v5; // r13
  void *v6; // r14
  const char *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // rax
  void *v12; // rdi
  __int64 v13; // r15
  _QWORD *v14; // r8
  _BYTE *v15; // rcx
  __int64 v16; // rax
  _BYTE *v17; // rax
  _QWORD *v18; // [rsp+0h] [rbp-160h]
  _QWORD *v19; // [rsp+8h] [rbp-158h]
  __int64 v20; // [rsp+18h] [rbp-148h] BYREF
  void *s2; // [rsp+20h] [rbp-140h] BYREF
  size_t n; // [rsp+28h] [rbp-138h]
  _BYTE v23[304]; // [rsp+30h] [rbp-130h] BYREF

  v2 = sub_16498A0(a1);
  if ( (unsigned __int8)sub_16033A0(v2) && *(_BYTE *)(a1 + 16) > 3u )
    return;
  v3 = *((_BYTE *)a2 + 16);
  if ( v3 > 1u )
  {
    v4 = *((_BYTE *)a2 + 17) == 1;
    s2 = v23;
    n = 0x10000000000LL;
    if ( !v4 )
    {
LABEL_6:
      sub_16E2F40(a2, &s2);
      v5 = (unsigned int)n;
      v6 = s2;
      goto LABEL_7;
    }
  }
  else
  {
    if ( (*(_BYTE *)(a1 + 23) & 0x20) == 0 )
      return;
    v4 = *((_BYTE *)a2 + 17) == 1;
    s2 = v23;
    n = 0x10000000000LL;
    if ( !v4 )
      goto LABEL_6;
    if ( v3 == 1 )
    {
      v5 = 0;
      v6 = 0;
      goto LABEL_7;
    }
  }
  v13 = *a2;
  switch ( v3 )
  {
    case 3u:
      v5 = 0;
      if ( v13 )
        v5 = strlen((const char *)v13);
      v6 = (void *)v13;
      break;
    case 4u:
    case 5u:
      v6 = *(void **)v13;
      v5 = *(_QWORD *)(v13 + 8);
      break;
    case 6u:
      v5 = *(unsigned int *)(v13 + 8);
      v6 = *(void **)v13;
      break;
    default:
      goto LABEL_6;
  }
LABEL_7:
  v7 = sub_1649960(a1);
  if ( v5 != v8 || v5 && memcmp(v7, v6, v5) )
  {
    if ( (unsigned int)dword_4F9F020 < v5 && *(_BYTE *)(a1 + 16) > 3u )
    {
      if ( (unsigned int)dword_4F9F020 > 1 )
      {
        if ( v5 > (unsigned int)dword_4F9F020 )
          v5 = (unsigned int)dword_4F9F020;
      }
      else
      {
        v5 = 1;
      }
    }
    if ( !(unsigned __int8)sub_1648C30(a1, &v20) )
    {
      v9 = v20;
      if ( v20 )
      {
        if ( (*(_BYTE *)(a1 + 23) & 0x20) == 0 )
          goto LABEL_15;
        v10 = sub_16498B0(a1);
        sub_164D860(v9, v10);
        sub_164B400(a1);
        if ( v5 )
        {
          v9 = v20;
LABEL_15:
          v11 = sub_164D870(v9, v6, v5, a1);
          sub_164B0D0(a1, v11);
          v12 = s2;
          if ( s2 == v23 )
            return;
LABEL_19:
          _libc_free((unsigned __int64)v12);
          return;
        }
        goto LABEL_18;
      }
      if ( v5 )
      {
        sub_164B400(a1);
        v14 = (_QWORD *)malloc(v5 + 17);
        if ( !v14 )
        {
          if ( v5 == -17 )
          {
            v16 = malloc(1u);
            v14 = 0;
            if ( v16 )
            {
              v15 = (_BYTE *)(v16 + 16);
              v14 = (_QWORD *)v16;
              goto LABEL_43;
            }
          }
          v18 = v14;
          sub_16BD1C0("Allocation failed");
          v14 = v18;
        }
        v15 = v14 + 2;
        if ( v5 + 1 <= 1 )
        {
LABEL_32:
          v15[v5] = 0;
          *v14 = v5;
          v14[1] = 0;
          sub_164B0D0(a1, (__int64)v14);
          *(_QWORD *)(sub_16498B0(a1) + 8) = a1;
          goto LABEL_18;
        }
LABEL_43:
        v19 = v14;
        v17 = memcpy(v15, v6, v5);
        v14 = v19;
        v15 = v17;
        goto LABEL_32;
      }
      sub_164B400(a1);
    }
  }
LABEL_18:
  v12 = s2;
  if ( s2 != v23 )
    goto LABEL_19;
}
