// Function: sub_25975F0
// Address: 0x25975f0
//
_BOOL8 __fastcall sub_25975F0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v5; // rsi
  int v6; // ebx
  __int64 v7; // rdi
  __int64 (__fastcall *v8)(__int64, __int64 (__fastcall *)(__int64, unsigned __int64), __int64, char); // rax
  char v9; // r9
  unsigned __int8 **v10; // rax
  unsigned __int8 **v11; // rdi
  int *v12; // r10
  __int64 v13; // r8
  unsigned __int8 *v14; // rcx
  int v15; // esi
  __int64 v16; // r11
  unsigned int v17; // edx
  int v18; // r13d
  __int64 v19; // r14
  unsigned int v20; // esi
  int v21; // ecx
  char v23; // dl
  unsigned __int8 *v24; // rax
  unsigned int v25; // r14d
  __int64 v26; // rcx
  __int64 v27; // r14
  int v28; // ecx
  bool v29; // dl
  __int64 v30; // r15
  unsigned int v31; // r14d
  __int64 v32; // r14
  int v33; // [rsp+Ch] [rbp-44h] BYREF
  int *v34; // [rsp+10h] [rbp-40h] BYREF
  __int64 v35; // [rsp+18h] [rbp-38h]

  v34 = (int *)sub_250ED40(*(_QWORD *)(a2 + 208));
  if ( !BYTE4(v34) )
    abort();
  v3 = *(_QWORD *)(a1 + 80);
  v5 = *(_QWORD *)(a1 + 72);
  v33 = (int)v34;
  v6 = *(_DWORD *)(a1 + 100);
  v34 = &v33;
  v35 = a1;
  v7 = sub_252AE70(a2, v5, v3, a1, 0, 0, 1);
  v8 = *(__int64 (__fastcall **)(__int64, __int64 (__fastcall *)(__int64, unsigned __int64), __int64, char))(*(_QWORD *)v7 + 112LL);
  if ( v8 == sub_254E4A0 )
  {
    v9 = *(_BYTE *)(v7 + 97);
    if ( v9 )
    {
      v10 = *(unsigned __int8 ***)(v7 + 248);
      v11 = &v10[*(unsigned int *)(v7 + 256)];
      if ( v10 != v11 )
      {
        v12 = v34;
        v13 = v35;
        while ( 1 )
        {
          v14 = *v10;
          v15 = **v10;
          if ( (unsigned int)(v15 - 12) <= 1 )
            goto LABEL_6;
          v16 = *((_QWORD *)v14 + 1);
          v17 = *(unsigned __int8 *)(v16 + 8) - 17;
          if ( (_BYTE)v15 != 22 )
          {
            v21 = *(_DWORD *)(v13 + 100);
LABEL_16:
            if ( v17 > 1 )
            {
              v25 = *(_DWORD *)(v16 + 8);
LABEL_36:
              v20 = v25 >> 8;
LABEL_12:
              if ( v21 == -1 )
                goto LABEL_19;
            }
            else
            {
              v19 = *(_QWORD *)(v16 + 16);
LABEL_18:
              v20 = *(_DWORD *)(*(_QWORD *)v19 + 8LL) >> 8;
              if ( v21 == -1 )
                goto LABEL_19;
            }
            if ( v21 != v20 )
              goto LABEL_14;
LABEL_6:
            if ( v11 == ++v10 )
              goto LABEL_20;
            continue;
          }
          v18 = *v12;
          if ( v17 > 1 )
          {
            v25 = *(_DWORD *)(v16 + 8);
            v20 = v25 >> 8;
            if ( v25 >> 8 != v18 )
            {
LABEL_11:
              v21 = *(_DWORD *)(v13 + 100);
              goto LABEL_12;
            }
            v26 = *((_QWORD *)v14 + 2);
            if ( !v26 )
            {
              v21 = *(_DWORD *)(v13 + 100);
              goto LABEL_36;
            }
          }
          else
          {
            v19 = *(_QWORD *)(v16 + 16);
            v20 = *(_DWORD *)(*(_QWORD *)v19 + 8LL) >> 8;
            if ( v20 != v18 )
              goto LABEL_11;
            v26 = *((_QWORD *)v14 + 2);
            if ( !v26 )
            {
              v21 = *(_DWORD *)(v13 + 100);
              goto LABEL_18;
            }
          }
          v20 = *v12;
          do
          {
            v27 = *(_QWORD *)(v26 + 24);
            if ( *(_BYTE *)v27 != 79 )
            {
              if ( v17 <= 1 )
                v16 = **(_QWORD **)(v16 + 16);
              v28 = *(_DWORD *)(v13 + 100);
              v20 = *(_DWORD *)(v16 + 8) >> 8;
              v29 = v20 == v28;
              if ( v28 == -1 )
                goto LABEL_19;
LABEL_33:
              if ( v29 )
                goto LABEL_6;
              *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
              return 0;
            }
            v30 = *(_QWORD *)(v27 + 8);
            v31 = *(unsigned __int8 *)(v30 + 8) - 17;
            if ( v20 == v18 )
            {
              if ( v31 <= 1 )
              {
                v32 = *(_QWORD *)(v30 + 16);
LABEL_41:
                v20 = *(_DWORD *)(*(_QWORD *)v32 + 8LL) >> 8;
                goto LABEL_42;
              }
              v20 = *(_DWORD *)(v30 + 8) >> 8;
            }
            else
            {
              if ( v31 <= 1 )
              {
                v32 = *(_QWORD *)(v30 + 16);
                if ( *(_DWORD *)(*(_QWORD *)v32 + 8LL) >> 8 != v20 )
                  goto LABEL_14;
                goto LABEL_41;
              }
              if ( *(_DWORD *)(v30 + 8) >> 8 != v20 )
                goto LABEL_14;
            }
LABEL_42:
            v26 = *(_QWORD *)(v26 + 8);
          }
          while ( v26 );
          v21 = *(_DWORD *)(v13 + 100);
          if ( v20 == v18 )
            goto LABEL_16;
          v29 = v20 == v21;
          if ( v21 != -1 )
            goto LABEL_33;
LABEL_19:
          ++v10;
          *(_DWORD *)(v13 + 100) = v20;
          if ( v11 == v10 )
          {
LABEL_20:
            v23 = v9;
            goto LABEL_22;
          }
        }
      }
      return *(_DWORD *)(a1 + 100) == v6;
    }
    v24 = (unsigned __int8 *)sub_250D070((_QWORD *)(v7 + 72));
    v23 = sub_25350C0(&v34, v24);
  }
  else
  {
    v23 = v8(v7, (__int64 (__fastcall *)(__int64, unsigned __int64))sub_25350C0, (__int64)&v34, 2);
  }
LABEL_22:
  if ( !v23 )
  {
LABEL_14:
    *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
    return 0;
  }
  return *(_DWORD *)(a1 + 100) == v6;
}
