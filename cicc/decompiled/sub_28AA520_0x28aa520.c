// Function: sub_28AA520
// Address: 0x28aa520
//
__int64 __fastcall sub_28AA520(__int64 a1, __int64 a2, unsigned __int8 (__fastcall *a3)(__int64, __int64), __int64 a4)
{
  unsigned int v5; // eax
  __int64 v6; // r8
  __int64 v7; // r9
  unsigned int v8; // r14d
  unsigned int v9; // eax
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 *v12; // r15
  __int64 v13; // rbx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rdx
  __int64 **v17; // rax
  unsigned int v18; // r12d
  char v19; // dl
  unsigned __int16 v20; // r13
  __int64 v21; // r8
  __int64 v22; // r9
  unsigned int v23; // eax
  __int64 v24; // r8
  __int64 v25; // r9
  char v26; // al
  __int64 *v27; // rcx
  __int64 v28; // rcx
  unsigned int v29; // esi
  __int64 *v30; // rdx
  __int64 v31; // rdx
  _BYTE *v32; // rdi
  __int64 *v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // rdi
  __int64 *v38; // rax
  __int64 v39; // rax
  unsigned __int64 v40; // rdx
  __int64 v41; // r13
  __int64 v42; // rax
  _QWORD *v47; // [rsp+30h] [rbp-1A0h] BYREF
  __int64 v48; // [rsp+38h] [rbp-198h]
  _QWORD v49[8]; // [rsp+40h] [rbp-190h] BYREF
  __int64 v50; // [rsp+80h] [rbp-150h] BYREF
  __int64 **v51; // [rsp+88h] [rbp-148h]
  __int64 v52; // [rsp+90h] [rbp-140h]
  int v53; // [rsp+98h] [rbp-138h]
  unsigned __int8 v54; // [rsp+9Ch] [rbp-134h]
  char v55; // [rsp+A0h] [rbp-130h] BYREF

  v47 = v49;
  v49[0] = a2;
  v48 = 0x800000001LL;
  v5 = sub_D138F0();
  v8 = v5;
  if ( v5 > 8 )
    sub_C8D5F0((__int64)&v47, v49, v5, 8u, v6, v7);
  v50 = 0;
  v51 = (__int64 **)&v55;
  v9 = v48;
  v52 = 32;
  v53 = 0;
  v54 = 1;
  if ( (_DWORD)v48 )
  {
    do
    {
      v10 = v9--;
      v11 = v47[v10 - 1];
      LODWORD(v48) = v9;
      v12 = *(__int64 **)(v11 + 16);
      if ( v12 )
      {
        while ( 1 )
        {
          v13 = v12[3];
          if ( !(unsigned __int8)sub_B19DB0(*(_QWORD *)(*(_QWORD *)a1 + 24LL), **(_QWORD **)(a1 + 8), v13) )
            **(_BYTE **)(a1 + 16) = 1;
          v16 = v54;
          if ( v8 <= HIDWORD(v52) - v53 )
          {
LABEL_51:
            v18 = 0;
            goto LABEL_52;
          }
          if ( v54 )
          {
            v17 = v51;
            v16 = (__int64)&v51[HIDWORD(v52)];
            if ( v51 != (__int64 **)v16 )
            {
              while ( *v17 != v12 )
              {
                if ( (__int64 **)v16 == ++v17 )
                  goto LABEL_41;
              }
              goto LABEL_13;
            }
LABEL_41:
            if ( HIDWORD(v52) < (unsigned int)v52 )
              break;
          }
          sub_C8CC70((__int64)&v50, (__int64)v12, v16, HIDWORD(v52), v14, v15);
          if ( v19 )
          {
LABEL_18:
            v20 = sub_D139D0(
                    v12,
                    a2,
                    (unsigned __int8 (__fastcall *)(__int64, unsigned __int8 *, __int64))sub_28A94F0,
                    *(_QWORD *)(a1 + 24));
            if ( (_BYTE)v20 )
              goto LABEL_51;
            if ( !(unsigned __int8)sub_B46420(v13) && !(unsigned __int8)sub_B46490(v13) )
              goto LABEL_36;
            LOBYTE(v23) = sub_B46A10(v13);
            v25 = v23;
            v26 = *(_BYTE *)(v13 + 7);
            if ( (_BYTE)v25 )
            {
              if ( (v26 & 0x40) != 0 )
                v27 = *(__int64 **)(v13 - 8);
              else
                v27 = (__int64 *)(v13 - 32LL * (*(_DWORD *)(v13 + 4) & 0x7FFFFFF));
              v28 = *v27;
              v29 = *(_DWORD *)(v28 + 32);
              v30 = *(__int64 **)(v28 + 24);
              if ( v29 > 0x40 )
              {
                v31 = *v30;
                if ( v31 < 0 )
                  goto LABEL_44;
              }
              else if ( v29 )
              {
                v31 = (__int64)((_QWORD)v30 << (64 - (unsigned __int8)v29)) >> (64 - (unsigned __int8)v29);
                if ( v31 < 0 )
                  goto LABEL_44;
              }
              else
              {
                v31 = 0;
              }
              v32 = *(_BYTE **)(a1 + 32);
              if ( !v32[16] )
                goto LABEL_28;
              if ( v31 != sub_CA1930(v32) )
              {
                v26 = *(_BYTE *)(v13 + 7);
                goto LABEL_28;
              }
LABEL_44:
              v41 = *(_QWORD *)(a1 + 40);
              v42 = *(unsigned int *)(v41 + 8);
              if ( v42 + 1 > (unsigned __int64)*(unsigned int *)(v41 + 12) )
              {
                sub_C8D5F0(*(_QWORD *)(a1 + 40), (const void *)(v41 + 16), v42 + 1, 8u, v24, v25);
                v42 = *(unsigned int *)(v41 + 8);
              }
              *(_QWORD *)(*(_QWORD *)v41 + 8 * v42) = v13;
              ++*(_DWORD *)(v41 + 8);
              v12 = (__int64 *)v12[1];
              if ( !v12 )
              {
LABEL_14:
                v9 = v48;
                goto LABEL_15;
              }
            }
            else
            {
LABEL_28:
              if ( (v26 & 0x20) != 0 && sub_B91C10(v13, 8) )
              {
                v37 = *(_QWORD *)(a1 + 48);
                if ( !*(_BYTE *)(v37 + 28) )
                  goto LABEL_50;
                v38 = *(__int64 **)(v37 + 8);
                v34 = *(unsigned int *)(v37 + 20);
                v33 = &v38[v34];
                if ( v38 == v33 )
                {
LABEL_49:
                  if ( (unsigned int)v34 >= *(_DWORD *)(v37 + 16) )
                  {
LABEL_50:
                    sub_C8CC70(v37, v13, (__int64)v33, v34, v35, v36);
                    goto LABEL_35;
                  }
                  *(_DWORD *)(v37 + 20) = v34 + 1;
                  *v33 = v13;
                  ++*(_QWORD *)v37;
                }
                else
                {
                  while ( v13 != *v38 )
                  {
                    if ( v33 == ++v38 )
                      goto LABEL_49;
                  }
                }
              }
LABEL_35:
              if ( !a3(a4, v13) )
                goto LABEL_51;
LABEL_36:
              if ( !HIBYTE(v20) )
                goto LABEL_13;
              v39 = (unsigned int)v48;
              v40 = (unsigned int)v48 + 1LL;
              if ( v40 > HIDWORD(v48) )
              {
                sub_C8D5F0((__int64)&v47, v49, v40, 8u, v21, v22);
                v39 = (unsigned int)v48;
              }
              v47[v39] = v13;
              LODWORD(v48) = v48 + 1;
              v12 = (__int64 *)v12[1];
              if ( !v12 )
                goto LABEL_14;
            }
          }
          else
          {
LABEL_13:
            v12 = (__int64 *)v12[1];
            if ( !v12 )
              goto LABEL_14;
          }
        }
        ++HIDWORD(v52);
        *(_QWORD *)v16 = v12;
        ++v50;
        goto LABEL_18;
      }
LABEL_15:
      ;
    }
    while ( v9 );
    v18 = 1;
LABEL_52:
    if ( !v54 )
      _libc_free((unsigned __int64)v51);
  }
  else
  {
    v18 = 1;
  }
  if ( v47 != v49 )
    _libc_free((unsigned __int64)v47);
  return v18;
}
