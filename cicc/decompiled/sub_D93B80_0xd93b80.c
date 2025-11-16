// Function: sub_D93B80
// Address: 0xd93b80
//
__int64 __fastcall sub_D93B80(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v6; // rdx
  unsigned int v7; // ecx
  int v8; // r14d
  __int64 v9; // rdx
  __int64 v10; // r12
  bool v11; // cc
  __int64 *v12; // rcx
  __int64 v13; // rcx
  unsigned __int8 *v14; // rdi
  unsigned __int8 *v15; // r8
  __int64 v16; // rax
  unsigned __int8 v17; // dl
  __int64 v18; // r8
  unsigned __int8 *v19; // rdi
  __int64 v20; // r9
  __int64 v21; // rdi
  int v22; // eax
  __int64 v23; // rdi
  int v24; // eax
  __int64 v25; // rdi
  int v26; // eax
  __int64 v27; // rdi
  int v28; // eax
  __int64 v29; // rdx
  __int64 v30; // [rsp+0h] [rbp-60h]
  __int64 v31; // [rsp+0h] [rbp-60h]
  unsigned __int8 v32; // [rsp+Fh] [rbp-51h]
  unsigned __int8 v33; // [rsp+Fh] [rbp-51h]
  unsigned __int8 v34; // [rsp+Fh] [rbp-51h]
  char v35; // [rsp+Fh] [rbp-51h]
  __int64 v36; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v37; // [rsp+18h] [rbp-48h]
  __int64 v38; // [rsp+20h] [rbp-40h] BYREF
  int v39; // [rsp+28h] [rbp-38h]

  *(_QWORD *)a1 = 0;
  *(_DWORD *)(a1 + 16) = 1;
  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 32) = 1;
  *(_QWORD *)(a1 + 24) = 0;
  v37 = a2;
  if ( a2 > 0x40 )
    sub_C43690((__int64)&v36, 0, 0);
  else
    v36 = 0;
  result = *(unsigned __int16 *)(a3 + 24);
  if ( (_WORD)result != 5 )
    goto LABEL_14;
  if ( *(_QWORD *)(a3 + 40) == 2 )
  {
    result = *(_QWORD *)(a3 + 32);
    if ( !*(_WORD *)(*(_QWORD *)result + 24LL) )
    {
      v6 = *(_QWORD *)(*(_QWORD *)result + 32LL);
      if ( v37 <= 0x40 && (v7 = *(_DWORD *)(v6 + 32), v7 <= 0x40) )
      {
        v29 = *(_QWORD *)(v6 + 24);
        v37 = v7;
        v36 = v29;
      }
      else
      {
        sub_C43990((__int64)&v36, v6 + 24);
        result = *(_QWORD *)(a3 + 32);
      }
      a3 = *(_QWORD *)(result + 8);
      result = *(unsigned __int16 *)(a3 + 24);
LABEL_14:
      if ( (unsigned __int16)(result - 2) > 2u )
      {
        v9 = 0;
        v8 = 0;
      }
      else
      {
        a3 = *(_QWORD *)(a3 + 32);
        v8 = (unsigned __int16)result;
        v9 = 1;
        result = *(unsigned __int16 *)(a3 + 24);
      }
      if ( (_WORD)result == 15 )
      {
        v10 = *(_QWORD *)(a3 - 8);
        if ( *(_BYTE *)v10 == 86 )
        {
          v12 = (*(_BYTE *)(v10 + 7) & 0x40) != 0
              ? *(__int64 **)(v10 - 8)
              : (__int64 *)(v10 - 32LL * (*(_DWORD *)(v10 + 4) & 0x7FFFFFF));
          result = *v12;
          if ( *v12 )
          {
            *(_QWORD *)a1 = result;
            if ( (*(_BYTE *)(v10 + 7) & 0x40) != 0 )
              v13 = *(_QWORD *)(v10 - 8);
            else
              v13 = v10 - 32LL * (*(_DWORD *)(v10 + 4) & 0x7FFFFFF);
            v14 = *(unsigned __int8 **)(v13 + 32);
            result = *v14;
            v15 = v14 + 24;
            if ( (_BYTE)result == 17 )
              goto LABEL_28;
            v35 = v9;
            if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v14 + 1) + 8LL) - 17 <= 1
              && (unsigned __int8)result <= 0x15u )
            {
              result = (__int64)sub_AD7630((__int64)v14, 0, v9);
              if ( result )
              {
                if ( *(_BYTE *)result == 17 )
                {
                  LOBYTE(v9) = v35;
                  v15 = (unsigned __int8 *)(result + 24);
LABEL_28:
                  v30 = (__int64)v15;
                  v32 = v9;
                  v16 = sub_986520(v10);
                  v17 = v32;
                  v18 = v30;
                  v19 = *(unsigned __int8 **)(v16 + 64);
                  result = *v19;
                  v20 = (__int64)(v19 + 24);
                  if ( (_BYTE)result == 17 )
                  {
LABEL_29:
                    if ( *(_DWORD *)(a1 + 16) <= 0x40u && *(_DWORD *)(v18 + 8) <= 0x40u )
                    {
                      *(_QWORD *)(a1 + 8) = *(_QWORD *)v18;
                      *(_DWORD *)(a1 + 16) = *(_DWORD *)(v18 + 8);
                    }
                    else
                    {
                      v31 = v20;
                      v33 = v17;
                      sub_C43990(a1 + 8, v18);
                      v20 = v31;
                      v17 = v33;
                    }
                    if ( *(_DWORD *)(a1 + 32) <= 0x40u && *(_DWORD *)(v20 + 8) <= 0x40u )
                    {
                      *(_QWORD *)(a1 + 24) = *(_QWORD *)v20;
                      *(_DWORD *)(a1 + 32) = *(_DWORD *)(v20 + 8);
                    }
                    else
                    {
                      v34 = v17;
                      sub_C43990(a1 + 24, v20);
                      v17 = v34;
                    }
                    if ( !v17 )
                    {
LABEL_45:
                      sub_C45EE0(a1 + 8, &v36);
                      result = sub_C45EE0(a1 + 24, &v36);
                      if ( v37 <= 0x40 )
                        return result;
                      goto LABEL_6;
                    }
                    switch ( v8 )
                    {
                      case 3:
                        sub_C449B0((__int64)&v38, (const void **)(a1 + 8), a2);
                        if ( *(_DWORD *)(a1 + 16) > 0x40u )
                        {
                          v27 = *(_QWORD *)(a1 + 8);
                          if ( v27 )
                            j_j___libc_free_0_0(v27);
                        }
                        *(_QWORD *)(a1 + 8) = v38;
                        v28 = v39;
                        v39 = 0;
                        *(_DWORD *)(a1 + 16) = v28;
                        sub_969240(&v38);
                        sub_C449B0((__int64)&v38, (const void **)(a1 + 24), a2);
                        if ( *(_DWORD *)(a1 + 32) <= 0x40u )
                          goto LABEL_44;
                        break;
                      case 4:
                        sub_C44830((__int64)&v38, (_DWORD *)(a1 + 8), a2);
                        if ( *(_DWORD *)(a1 + 16) > 0x40u )
                        {
                          v21 = *(_QWORD *)(a1 + 8);
                          if ( v21 )
                            j_j___libc_free_0_0(v21);
                        }
                        *(_QWORD *)(a1 + 8) = v38;
                        v22 = v39;
                        v39 = 0;
                        *(_DWORD *)(a1 + 16) = v22;
                        sub_969240(&v38);
                        sub_C44830((__int64)&v38, (_DWORD *)(a1 + 24), a2);
                        if ( *(_DWORD *)(a1 + 32) <= 0x40u )
                          goto LABEL_44;
                        break;
                      case 2:
                        sub_C44740((__int64)&v38, (char **)(a1 + 8), a2);
                        if ( *(_DWORD *)(a1 + 16) > 0x40u )
                        {
                          v25 = *(_QWORD *)(a1 + 8);
                          if ( v25 )
                            j_j___libc_free_0_0(v25);
                        }
                        *(_QWORD *)(a1 + 8) = v38;
                        v26 = v39;
                        v39 = 0;
                        *(_DWORD *)(a1 + 16) = v26;
                        sub_969240(&v38);
                        sub_C44740((__int64)&v38, (char **)(a1 + 24), a2);
                        if ( *(_DWORD *)(a1 + 32) <= 0x40u )
                          goto LABEL_44;
                        break;
                      default:
                        BUG();
                    }
                    v23 = *(_QWORD *)(a1 + 24);
                    if ( v23 )
                      j_j___libc_free_0_0(v23);
LABEL_44:
                    *(_QWORD *)(a1 + 24) = v38;
                    v24 = v39;
                    v39 = 0;
                    *(_DWORD *)(a1 + 32) = v24;
                    sub_969240(&v38);
                    goto LABEL_45;
                  }
                  if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v19 + 1) + 8LL) - 17 <= 1
                    && (unsigned __int8)result <= 0x15u )
                  {
                    result = (__int64)sub_AD7630((__int64)v19, 0, v32);
                    if ( result )
                    {
                      if ( *(_BYTE *)result == 17 )
                      {
                        v18 = v30;
                        v17 = v32;
                        v20 = result + 24;
                        goto LABEL_29;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      v11 = v37 <= 0x40;
      *(_QWORD *)a1 = 0;
      if ( v11 )
        return result;
      goto LABEL_6;
    }
  }
  if ( v37 <= 0x40 )
    return result;
LABEL_6:
  if ( v36 )
    return j_j___libc_free_0_0(v36);
  return result;
}
