// Function: sub_3953170
// Address: 0x3953170
//
__int64 __fastcall sub_3953170(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r15
  char v6; // al
  __int64 v7; // rdx
  _QWORD *v8; // rax
  int v10; // r13d
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r12
  int v14; // r12d
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  unsigned __int64 v18; // r12
  __int64 v19; // rdi
  unsigned __int64 v20; // rsi
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // rdx
  char v23; // cl
  size_t v24; // r9
  int v25; // edx
  __int64 v26; // rax
  __int64 v27; // rax
  unsigned int v28; // r9d
  int v29; // edx
  int v30; // [rsp+0h] [rbp-80h]
  char v31; // [rsp+7h] [rbp-79h]
  __int64 *v32; // [rsp+8h] [rbp-78h]
  char *v33; // [rsp+10h] [rbp-70h] BYREF
  size_t v34; // [rsp+18h] [rbp-68h]
  _QWORD v35[2]; // [rsp+20h] [rbp-60h] BYREF
  char *v36; // [rsp+30h] [rbp-50h] BYREF
  size_t v37; // [rsp+38h] [rbp-48h]
  _QWORD v38[8]; // [rsp+40h] [rbp-40h] BYREF

  if ( *(_BYTE *)(a3 + 16) != 78 )
    goto LABEL_9;
  v4 = *(_QWORD *)(a3 - 24);
  v6 = *(_BYTE *)(v4 + 16);
  if ( !v6 )
  {
    if ( (*(_BYTE *)(v4 + 33) & 0x20) != 0 && (unsigned int)(*(_DWORD *)(v4 + 36) - 4386) <= 1 )
    {
      v7 = *(_QWORD *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF));
      v8 = *(_QWORD **)(v7 + 24);
      if ( *(_DWORD *)(v7 + 32) > 0x40u )
        v8 = (_QWORD *)*v8;
      *(_BYTE *)(a1 + 4) = 1;
      *(_DWORD *)a1 = (_DWORD)v8;
      return a1;
    }
LABEL_9:
    *(_BYTE *)(a1 + 4) = 0;
    return a1;
  }
  if ( v6 != 20 )
    goto LABEL_9;
  v32 = (__int64 *)(v4 + 24);
  v10 = *(_DWORD *)(a3 + 20) & 0xFFFFFFF;
  if ( *(char *)(a3 + 23) < 0 )
  {
    v11 = sub_1648A40(a3);
    v13 = v11 + v12;
    if ( *(char *)(a3 + 23) >= 0 )
    {
      if ( !(unsigned int)(v13 >> 4) )
        goto LABEL_17;
    }
    else
    {
      if ( !(unsigned int)((v13 - sub_1648A40(a3)) >> 4) )
        goto LABEL_17;
      if ( *(char *)(a3 + 23) < 0 )
      {
        v14 = *(_DWORD *)(sub_1648A40(a3) + 8);
        if ( *(char *)(a3 + 23) >= 0 )
          BUG();
        v15 = sub_1648A40(a3);
        v10 += v14 - *(_DWORD *)(v15 + v16 - 4);
        goto LABEL_17;
      }
    }
    BUG();
  }
LABEL_17:
  if ( v10 == 2 && (v26 = *(_QWORD *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF)), *(_BYTE *)(v26 + 16) == 13) )
  {
    if ( *(_DWORD *)(v26 + 32) <= 0x40u )
      v27 = *(_QWORD *)(v26 + 24);
    else
      v27 = **(_QWORD **)(v26 + 24);
    v30 = v27;
    v31 = 1;
  }
  else
  {
    v31 = 0;
  }
  v33 = (char *)v35;
  sub_3952BD0((__int64 *)&v33, "setmaxnreg.", (__int64)"");
  v17 = sub_22416F0(v32, v33, 0, v34);
  if ( v17 != -1 )
  {
    v36 = (char *)v38;
    v18 = v17 + v34 + 3;
    sub_3952BD0((__int64 *)&v36, ".sync.aligned.u32", (__int64)"");
    if ( v18 == sub_22416F0(v32, v36, v18, v37) )
    {
      v19 = *(_QWORD *)(v4 + 24);
      v20 = *(_QWORD *)(v4 + 32);
      v21 = v18 + v37;
      if ( v31 )
      {
        if ( v21 >= v20 )
        {
          if ( v21 + 1 >= v20 || *(_BYTE *)(v19 + v21) != 36 || *(_BYTE *)(v19 + v21 + 1) != 48 )
          {
LABEL_32:
            *(_BYTE *)(a1 + 4) = 0;
            goto LABEL_33;
          }
          v22 = v18 + v37;
        }
        else
        {
          v22 = v18 + v37;
          while ( 1 )
          {
            v23 = *(_BYTE *)(v19 + v22);
            v24 = v22++;
            if ( v23 != 32 && v23 != 9 )
              break;
            if ( v20 <= v22 )
              goto LABEL_27;
          }
          v22 = v24;
LABEL_27:
          if ( v20 <= v22 + 1 || *(_BYTE *)(v19 + v22) != 36 || *(_BYTE *)(v19 + v22 + 1) != 48 )
          {
LABEL_29:
            while ( 1 )
            {
              LOBYTE(v25) = *(_BYTE *)(v19 + v21);
              if ( (_BYTE)v25 != 32 && (_BYTE)v25 != 9 )
                break;
              if ( v20 <= ++v21 )
                goto LABEL_32;
            }
            v28 = 0;
            if ( (unsigned int)(unsigned __int8)v25 - 48 <= 9 )
            {
              while ( 1 )
              {
                v29 = (char)v25 - 48;
                if ( ~v29 / 0xAu < v28 )
                  break;
                ++v21;
                v28 = v29 + 10 * v28;
                if ( v20 > v21 )
                {
                  v25 = *(unsigned __int8 *)(v19 + v21);
                  if ( (unsigned int)(v25 - 48) <= 9 )
                    continue;
                }
                *(_BYTE *)(a1 + 4) = 1;
                *(_DWORD *)a1 = v28;
                goto LABEL_33;
              }
            }
            goto LABEL_32;
          }
        }
        if ( v20 <= v22 + 2 || (unsigned int)*(unsigned __int8 *)(v19 + v22 + 2) - 48 > 9 )
        {
          *(_BYTE *)(a1 + 4) = 1;
          *(_DWORD *)a1 = v30;
LABEL_33:
          if ( v36 != (char *)v38 )
            j_j___libc_free_0((unsigned __int64)v36);
          goto LABEL_38;
        }
      }
      if ( v21 < v20 )
        goto LABEL_29;
      goto LABEL_32;
    }
    if ( v36 != (char *)v38 )
      j_j___libc_free_0((unsigned __int64)v36);
  }
  *(_BYTE *)(a1 + 4) = 0;
LABEL_38:
  if ( v33 != (char *)v35 )
    j_j___libc_free_0((unsigned __int64)v33);
  return a1;
}
