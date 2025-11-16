// Function: sub_1D0DB10
// Address: 0x1d0db10
//
unsigned __int64 __fastcall sub_1D0DB10(_QWORD *a1)
{
  __int64 (*v1)(void); // rax
  __int64 v2; // r14
  unsigned __int64 result; // rax
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // rax
  unsigned int v8; // r11d
  unsigned int *v9; // rsi
  __int64 v10; // r15
  __int16 v11; // cx
  unsigned __int64 v12; // rdx
  int v13; // eax
  unsigned int *v14; // rax
  unsigned int i; // ebx
  __int64 v16; // r12
  __int64 v17; // rdx
  char v18; // r8
  __int64 v19; // rsi
  unsigned int v20; // r9d
  __int64 v21; // rax
  unsigned int v22; // ecx
  unsigned __int64 v23; // rax
  int v24; // edx
  void (*v25)(); // rax
  unsigned __int16 v26; // ax
  __int64 v27; // rax
  bool v28; // zf
  __int64 v29; // rax
  unsigned int v30; // ecx
  __int64 v31; // rdx
  __int64 v32; // [rsp+8h] [rbp-78h]
  unsigned __int64 v33; // [rsp+10h] [rbp-70h]
  char v34; // [rsp+1Eh] [rbp-62h]
  char v35; // [rsp+1Fh] [rbp-61h]
  __int64 v36; // [rsp+20h] [rbp-60h]
  __int64 v37; // [rsp+30h] [rbp-50h]
  __int64 v38; // [rsp+38h] [rbp-48h]
  __int64 v39; // [rsp+40h] [rbp-40h] BYREF
  int v40; // [rsp+48h] [rbp-38h]
  _BOOL4 v41; // [rsp+4Ch] [rbp-34h]

  v35 = 0;
  v36 = *(_QWORD *)(a1[4] + 16LL);
  v1 = *(__int64 (**)(void))(*a1 + 104LL);
  if ( v1 != sub_1CFBF50 )
    v35 = v1();
  v2 = a1[6];
  result = 0xF0F0F0F0F0F0F0F1LL * ((a1[7] - v2) >> 4);
  if ( (_DWORD)result )
  {
    v33 = 0;
    v32 = 272LL * (unsigned int)(result - 1);
    while ( 1 )
    {
      v38 = v33 + v2;
      v4 = *(_QWORD *)(v33 + v2);
      if ( *(__int16 *)(v4 + 24) < 0 )
      {
        v29 = *(_QWORD *)(a1[2] + 8LL) + ((__int64)~*(__int16 *)(v4 + 24) << 6);
        v30 = *(unsigned __int16 *)(v29 + 2);
        if ( *(_WORD *)(v29 + 2) )
        {
          v31 = 0;
          while ( v30 <= (unsigned int)v31 || (*(_BYTE *)(*(_QWORD *)(v29 + 40) + 8 * v31 + 4) & 1) == 0 )
          {
            if ( v30 == (_DWORD)++v31 )
              goto LABEL_51;
          }
          *(_BYTE *)(v38 + 228) |= 8u;
        }
LABEL_51:
        if ( (*(_BYTE *)(v29 + 10) & 0x20) != 0 )
          *(_BYTE *)(v38 + 228) |= 0x10u;
        goto LABEL_15;
      }
LABEL_6:
      v5 = *(unsigned int *)(v4 + 56);
      if ( (_DWORD)v5 )
        break;
LABEL_23:
      result = v33;
      if ( v32 == v33 )
        return result;
      v2 = a1[6];
      v33 += 272LL;
    }
LABEL_7:
    v37 = v5;
    v6 = 0;
    while ( 1 )
    {
      v7 = *(_QWORD *)(v4 + 32);
      v8 = v6;
      v9 = (unsigned int *)(v7 + 40 * v6);
      v10 = *(_QWORD *)v9;
      v11 = *(_WORD *)(*(_QWORD *)v9 + 24LL);
      v12 = (0x7FF0007FF22uLL >> v11) & 1;
      if ( (unsigned __int16)v11 >= 0x2Bu )
        LOBYTE(v12) = 0;
      if ( v11 == 209 )
        goto LABEL_12;
      if ( (_BYTE)v12 )
        goto LABEL_12;
      v16 = a1[6] + 272LL * *(int *)(v10 + 28);
      if ( v38 == v16 )
        goto LABEL_12;
      v17 = *(_QWORD *)(v10 + 40);
      v18 = *(_BYTE *)(v17 + 16LL * v9[2]);
      if ( v6 != 2 )
        break;
      if ( *(_WORD *)(v4 + 24) != 46 )
        break;
      v19 = *(unsigned int *)(*(_QWORD *)(v7 + 40) + 84LL);
      if ( (int)v19 < 0 )
        break;
      v20 = *(_DWORD *)(v7 + 88);
      if ( v11 == 47 )
      {
        if ( (_DWORD)v19 != *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v10 + 32) + 40LL) + 84LL) )
          break;
      }
      else
      {
        if ( v11 >= 0 )
          break;
        v21 = *(_QWORD *)(a1[2] + 8LL) + ((__int64)~v11 << 6);
        v22 = *(unsigned __int8 *)(v21 + 4);
        if ( v20 < v22 || (_DWORD)v19 != *(unsigned __int16 *)(*(_QWORD *)(v21 + 32) + 2LL * (v20 - v22)) )
          break;
      }
      if ( !(_DWORD)v19 )
        break;
      v34 = v18;
      v27 = sub_1F4ABE0(a1[3], v19, *(unsigned __int8 *)(v17 + 16LL * v20));
      v18 = v34;
      v8 = 2;
      if ( *(char *)(*(_QWORD *)v27 + 28LL) >= 0 )
        break;
      v23 = v16 & 0xFFFFFFFFFFFFFFF9LL;
      if ( v34 != 1 )
      {
LABEL_34:
        v24 = *(unsigned __int16 *)(v16 + 226);
        v39 = v23;
        v40 = v19;
        v41 = v24;
        if ( !v35 )
        {
          (*(void (__fastcall **)(_QWORD *, __int64, __int64, _QWORD))(*a1 + 72LL))(a1, v10, v4, v8);
          v25 = *(void (**)())(*(_QWORD *)v36 + 208LL);
          if ( v25 != nullsub_681 )
            ((void (__fastcall *)(__int64, __int64, __int64, __int64 *))v25)(v36, v16, v38, &v39);
        }
        goto LABEL_37;
      }
LABEL_45:
      v28 = *(_WORD *)(v10 + 24) == 2;
      v40 = 0;
      v39 = v23 | 6;
      v41 = !v28;
LABEL_37:
      if ( !(unsigned __int8)sub_1F01A00(v38, &v39, 1) && (v39 & 6) == 0 )
      {
        v26 = *(_WORD *)(v16 + 224);
        if ( v26 > 1u )
          *(_WORD *)(v16 + 224) = v26 - 1;
      }
LABEL_12:
      if ( ++v6 == v37 )
      {
        v13 = *(_DWORD *)(v4 + 56);
        if ( !v13 )
          goto LABEL_23;
        v14 = (unsigned int *)(*(_QWORD *)(v4 + 32) + 40LL * (unsigned int)(v13 - 1));
        v4 = *(_QWORD *)v14;
        if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v14 + 40LL) + 16LL * v14[2]) != 111 )
          goto LABEL_23;
LABEL_15:
        if ( *(__int16 *)(v4 + 24) < 0
          && *(_QWORD *)(*(_QWORD *)(a1[2] + 8LL) + ((__int64)~*(__int16 *)(v4 + 24) << 6) + 32) )
        {
          *(_BYTE *)(v38 + 228) |= 0x80u;
          for ( i = sub_1FE6580(v4); i; --i )
          {
            if ( (unsigned __int8)sub_1D18C40(v4) )
            {
              if ( i <= *(unsigned __int8 *)(*(_QWORD *)(a1[2] + 8LL)
                                           + ((unsigned __int64)(unsigned int)~*(__int16 *)(v4 + 24) << 6)
                                           + 4) )
                goto LABEL_6;
              *(_BYTE *)(v38 + 228) |= 0x40u;
              v5 = *(unsigned int *)(v4 + 56);
              if ( (_DWORD)v5 )
                goto LABEL_7;
              goto LABEL_23;
            }
          }
        }
        goto LABEL_6;
      }
    }
    LODWORD(v19) = 0;
    v23 = v16 & 0xFFFFFFFFFFFFFFF9LL;
    if ( v18 != 1 )
      goto LABEL_34;
    goto LABEL_45;
  }
  return result;
}
