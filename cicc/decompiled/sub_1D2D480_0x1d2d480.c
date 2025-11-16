// Function: sub_1D2D480
// Address: 0x1d2d480
//
__int64 __fastcall sub_1D2D480(__int64 a1, __int64 a2, unsigned int a3)
{
  size_t *v3; // r12
  __int16 v5; // ax
  __int64 v7; // rax
  size_t v8; // rdx
  _QWORD *v9; // rax
  bool v10; // zf
  _QWORD *v11; // rax
  unsigned __int8 *v12; // r12
  size_t v13; // rdx
  int v14; // eax
  __int64 v15; // rdx
  size_t **v16; // rax
  unsigned __int64 v17; // rdi
  const char *v18; // r15
  size_t v19; // rax
  size_t v20; // r8
  _QWORD *v21; // rdx
  int v22; // eax
  __int64 v23; // rsi
  int v24; // ecx
  __int64 v25; // rdi
  unsigned int v26; // edx
  __int64 *v27; // rax
  __int64 v28; // r8
  __int64 v29; // rax
  _QWORD *v30; // rdi
  __int64 v31; // rax
  size_t *v32; // rdx
  size_t *v33; // r13
  size_t *v34; // rdi
  __int64 v35; // rax
  __int64 v36; // rax
  int v37; // eax
  int v38; // r9d
  size_t na; // [rsp+8h] [rbp-78h]
  size_t n; // [rsp+8h] [rbp-78h]
  size_t v41; // [rsp+18h] [rbp-68h] BYREF
  _QWORD *v42; // [rsp+20h] [rbp-60h] BYREF
  size_t v43; // [rsp+28h] [rbp-58h]
  _QWORD v44[2]; // [rsp+30h] [rbp-50h] BYREF
  char v45; // [rsp+40h] [rbp-40h]

  v5 = *(_WORD *)(a2 + 24);
  if ( v5 > 41 )
  {
    LODWORD(v3) = 0;
    if ( v5 == 212 )
      return (unsigned int)v3;
  }
  else if ( v5 > 5 )
  {
    switch ( v5 )
    {
      case 6:
        v7 = *(unsigned __int8 *)(a2 + 88);
        v8 = *(_QWORD *)(a2 + 96);
        LOBYTE(v42) = v7;
        v43 = v8;
        if ( (_BYTE)v7 )
        {
          v9 = (_QWORD *)(*(_QWORD *)(a1 + 720) + 8 * v7);
          v10 = *v9 == 0;
          *v9 = 0;
          LOBYTE(v3) = !v10;
          return (unsigned int)v3;
        }
        v31 = sub_1D2D3A0(a1 + 744, (char *)&v42);
        v33 = v32;
        v3 = (size_t *)v31;
        n = *(_QWORD *)(a1 + 784);
        if ( v31 == *(_QWORD *)(a1 + 768) && v32 == (size_t *)(a1 + 752) )
        {
          sub_1D13CC0(*(_QWORD *)(a1 + 760));
          *(_QWORD *)(a1 + 760) = 0;
          *(_QWORD *)(a1 + 768) = v33;
          LOBYTE(v3) = n != 0;
          *(_QWORD *)(a1 + 776) = v33;
          *(_QWORD *)(a1 + 784) = 0;
        }
        else
        {
          if ( v32 == (size_t *)v31 )
            goto LABEL_24;
          do
          {
            v34 = v3;
            v3 = (size_t *)sub_220EF30(v3);
            v35 = sub_220F330(v34, a1 + 752);
            j_j___libc_free_0(v35, 56);
            v36 = *(_QWORD *)(a1 + 784) - 1LL;
            *(_QWORD *)(a1 + 784) = v36;
          }
          while ( v33 != v3 );
          LOBYTE(v3) = n != v36;
        }
        return (unsigned int)v3;
      case 7:
        v11 = (_QWORD *)(*(_QWORD *)(a1 + 696) + 8LL * *(unsigned int *)(a2 + 84));
        v10 = *v11 == 0;
        *v11 = 0;
        LOBYTE(v3) = !v10;
        return (unsigned int)v3;
      case 17:
        v12 = *(unsigned __int8 **)(a2 + 88);
        v13 = 0;
        if ( v12 )
          v13 = strlen(*(const char **)(a2 + 88));
        v14 = sub_16D1B30((__int64 *)(a1 + 792), v12, v13);
        if ( v14 == -1 )
          goto LABEL_24;
        v15 = *(_QWORD *)(a1 + 792);
        v16 = (size_t **)(v15 + 8LL * v14);
        if ( v16 == (size_t **)(v15 + 8LL * *(unsigned int *)(a1 + 800)) )
          goto LABEL_24;
        v3 = *v16;
        sub_16D1CB0(a1 + 792, *v16);
        v17 = (unsigned __int64)v3;
        LODWORD(v3) = 1;
        _libc_free(v17);
        return (unsigned int)v3;
      case 39:
        v18 = *(const char **)(a2 + 88);
        LODWORD(v3) = *(unsigned __int8 *)(a2 + 96);
        v42 = v44;
        if ( !v18 )
          sub_426248((__int64)"basic_string::_M_construct null not valid");
        v19 = strlen(v18);
        v41 = v19;
        v20 = v19;
        if ( v19 > 0xF )
        {
          na = v19;
          v29 = sub_22409D0(&v42, &v41, 0);
          v20 = na;
          v42 = (_QWORD *)v29;
          v30 = (_QWORD *)v29;
          v44[0] = v41;
        }
        else
        {
          if ( v19 == 1 )
          {
            LOBYTE(v44[0]) = *v18;
            v21 = v44;
            goto LABEL_19;
          }
          if ( !v19 )
          {
            v21 = v44;
            goto LABEL_19;
          }
          v30 = v44;
        }
        memcpy(v30, v18, v20);
        v19 = v41;
        v21 = v42;
LABEL_19:
        v43 = v19;
        *((_BYTE *)v21 + v19) = 0;
        v45 = (char)v3;
        LOBYTE(v3) = sub_1D2D2A0(a1 + 824, (__int64)&v42) != 0;
        if ( v42 != v44 )
          j_j___libc_free_0(v42, v44[0] + 1LL);
        return (unsigned int)v3;
      case 41:
        v22 = *(_DWORD *)(a1 + 896);
        LODWORD(v3) = 0;
        if ( !v22 )
          return (unsigned int)v3;
        v23 = *(_QWORD *)(a2 + 88);
        v24 = v22 - 1;
        v25 = *(_QWORD *)(a1 + 880);
        v26 = (v22 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
        v27 = (__int64 *)(v25 + 16LL * v26);
        v28 = *v27;
        if ( v23 == *v27 )
          goto LABEL_23;
        v37 = 1;
        break;
      default:
        return sub_16BDCA0(a1 + 320, (_QWORD *)a2, a3);
    }
    while ( v28 != -8 )
    {
      v38 = v37 + 1;
      v26 = v24 & (v37 + v26);
      v27 = (__int64 *)(v25 + 16LL * v26);
      v28 = *v27;
      if ( v23 == *v27 )
      {
LABEL_23:
        *v27 = -16;
        LODWORD(v3) = 1;
        --*(_DWORD *)(a1 + 888);
        ++*(_DWORD *)(a1 + 892);
        return (unsigned int)v3;
      }
      v37 = v38;
    }
LABEL_24:
    LODWORD(v3) = 0;
    return (unsigned int)v3;
  }
  return sub_16BDCA0(a1 + 320, (_QWORD *)a2, a3);
}
