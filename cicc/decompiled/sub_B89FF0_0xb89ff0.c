// Function: sub_B89FF0
// Address: 0xb89ff0
//
__int64 __fastcall sub_B89FF0(__int64 a1, unsigned __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // rsi
  __int64 i; // rax
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rax
  size_t v10; // rdx
  size_t v11; // r15
  const void *v12; // r14
  __int64 v13; // rsi
  __int64 v14; // rbx
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // r11
  unsigned __int64 v18; // rdi
  _QWORD *v19; // r8
  _QWORD *v20; // rax
  _QWORD *v21; // rsi
  unsigned __int8 v22; // al
  __int64 v23; // r8
  __int64 v24; // r12
  __int64 v25; // rbx
  _QWORD *v26; // rdi
  unsigned int v28; // eax
  unsigned int v29; // ecx
  unsigned __int64 v30; // r8
  unsigned __int64 v31; // rsi
  _QWORD *v32; // r9
  _QWORD *v33; // rax
  _QWORD *v34; // rdi
  __int64 v35; // [rsp+0h] [rbp-D0h]
  __int64 v36; // [rsp+8h] [rbp-C8h]
  __int64 *v37; // [rsp+10h] [rbp-C0h]
  unsigned int v38; // [rsp+18h] [rbp-B8h]
  unsigned int v39; // [rsp+1Ch] [rbp-B4h]
  __int64 v40; // [rsp+20h] [rbp-B0h]
  __int64 v41; // [rsp+20h] [rbp-B0h]
  unsigned int v42; // [rsp+20h] [rbp-B0h]
  __int64 v43; // [rsp+38h] [rbp-98h]
  unsigned __int8 v44; // [rsp+38h] [rbp-98h]
  unsigned __int8 v46; // [rsp+4Ah] [rbp-86h]
  char v47; // [rsp+4Bh] [rbp-85h]
  unsigned int v48; // [rsp+4Ch] [rbp-84h]
  __int64 v49; // [rsp+50h] [rbp-80h] BYREF
  __int64 v50; // [rsp+58h] [rbp-78h]
  __int64 v51; // [rsp+60h] [rbp-70h]
  _QWORD v52[12]; // [rsp+70h] [rbp-60h] BYREF

  v46 = 0;
  if ( !sub_B2FC80(a2) )
  {
    v3 = *(_QWORD *)(a1 + 184);
    v4 = a1 + 336;
    v37 = *(__int64 **)(a2 + 40);
    v5 = *(_QWORD *)(v3 + 8);
    for ( i = *(_QWORD *)(v3 + 16); v5 != i; *(_QWORD *)(v4 - 8) = v7 + 208 )
    {
      v7 = *(_QWORD *)(i - 8);
      i -= 8;
      v4 += 8;
    }
    v49 = 0;
    v51 = 0x1000000000LL;
    v50 = 0;
    v8 = sub_B6F970(*v37);
    v47 = (*(__int64 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v8 + 24LL))(v8, "size-info", 9);
    if ( v47 )
    {
      v38 = sub_B806A0(a1 + 176, (__int64)v37, (__int64)&v49);
      v39 = sub_B2BED0(a2);
    }
    else
    {
      v39 = 0;
    }
    v9 = sub_BD5D20(a2);
    v11 = v10;
    v12 = (const void *)v9;
    v35 = sub_C996C0("OptFunction", 11, v9, v10);
    v13 = *(unsigned int *)(a1 + 200);
    if ( (_DWORD)v13 )
    {
      v14 = a1 + 176;
      v46 = 0;
      v48 = 0;
      while ( 1 )
      {
        v15 = *(_QWORD *)(*(_QWORD *)(a1 + 192) + 8LL * v48);
        sub_B817B0(v14, v15, 0, 3, v12, v11);
        sub_B86470(v14, (__int64 *)v15);
        sub_B89740(v14, (__int64 *)v15);
        sub_C85EE0(v52);
        v52[2] = v15;
        v52[3] = a2;
        v52[0] = &unk_49DA748;
        v52[4] = 0;
        v16 = sub_BC4450(v15);
        v17 = v16;
        if ( v16 )
        {
          v43 = v16;
          sub_C9E250(v16);
          v17 = v43;
        }
        if ( !*(_BYTE *)(v15 + 168) )
          break;
        v18 = *(_QWORD *)(v15 + 64);
        v19 = *(_QWORD **)(*(_QWORD *)(v15 + 56) + 8 * (a2 % v18));
        if ( v19 )
        {
          v20 = (_QWORD *)*v19;
          if ( a2 == *(_QWORD *)(*v19 + 8LL) )
          {
LABEL_16:
            if ( *v19 )
              break;
          }
          else
          {
            while ( 1 )
            {
              v21 = (_QWORD *)*v20;
              if ( !*v20 )
                break;
              v19 = v20;
              if ( a2 % v18 != v21[1] % v18 )
                break;
              v20 = (_QWORD *)*v20;
              if ( a2 == v21[1] )
                goto LABEL_16;
            }
          }
        }
        v30 = *(_QWORD *)(a2 + 40);
        v31 = *(_QWORD *)(v15 + 120);
        v32 = *(_QWORD **)(*(_QWORD *)(v15 + 112) + 8 * (v30 % v31));
        if ( v32 )
        {
          v33 = (_QWORD *)*v32;
          if ( v30 == *(_QWORD *)(*v32 + 8LL) )
          {
LABEL_45:
            if ( *v32 )
              break;
          }
          else
          {
            while ( 1 )
            {
              v34 = (_QWORD *)*v33;
              if ( !*v33 )
                break;
              v32 = v33;
              if ( *(_QWORD *)(a2 + 40) % v31 != v34[1] % v31 )
                break;
              v33 = (_QWORD *)*v33;
              if ( v30 == v34[1] )
                goto LABEL_45;
            }
          }
        }
        v44 = 0;
LABEL_19:
        if ( v47 )
        {
          v41 = v17;
          v28 = sub_B2BED0(a2);
          v17 = v41;
          if ( v28 != v39 )
          {
            v36 = v41;
            v42 = v28;
            sub_B82CC0(v14, v15, (__int64)v37, v28 - (unsigned __int64)v39, v38, (__int64)&v49, a2);
            v17 = v36;
            v29 = v42 + v38 - v39;
            v39 = v42;
            v38 = v29;
          }
        }
        if ( v17 )
          sub_C9E2A0(v17);
        v52[0] = &unk_49DA748;
        nullsub_162(v52);
        if ( v44 )
        {
          sub_B817B0(v14, v15, 1, 3, v12, v11);
          sub_B865A0(v14, (__int64 *)v15);
          sub_B866C0(v14, (__int64 *)v15);
          nullsub_76();
          sub_B887D0(v14, (__int64 *)v15);
        }
        else
        {
          sub_B865A0(v14, (__int64 *)v15);
          sub_B866C0(v14, (__int64 *)v15);
          nullsub_76();
        }
        sub_B87180(v14, v15);
        v13 = v15;
        sub_B81BF0(v14, v15, v12, v11, 3);
        if ( ++v48 >= *(_DWORD *)(a1 + 200) )
          goto LABEL_25;
      }
      v40 = v17;
      v22 = (*(__int64 (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v15 + 144LL))(v15, a2);
      v17 = v40;
      v44 = v22;
      if ( v22 )
      {
        sub_B7FEB0(v15, a2);
        v17 = v40;
        v46 = v44;
      }
      goto LABEL_19;
    }
    v46 = 0;
LABEL_25:
    if ( v35 )
      sub_C9AF60(v35);
    v23 = v49;
    if ( HIDWORD(v50) && (_DWORD)v50 )
    {
      v24 = 8LL * (unsigned int)v50;
      v25 = 0;
      do
      {
        v26 = *(_QWORD **)(v23 + v25);
        if ( v26 != (_QWORD *)-8LL && v26 )
        {
          v13 = *v26 + 17LL;
          sub_C7D6A0(v26, v13, 8);
          v23 = v49;
        }
        v25 += 8;
      }
      while ( v24 != v25 );
    }
    _libc_free(v23, v13);
  }
  return v46;
}
