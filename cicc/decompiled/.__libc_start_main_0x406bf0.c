// Function: .__libc_start_main
// Address: 0x406bf0
//
// attributes: thunk
int __libc_start_main(
        int (*main)(int, char **, char **),
        int argc,
        char **ubp_av,
        void (*init)(),
        void (*fini)(),
        void (*rtld_fini)(),
        void *stack_end)
{
  return _libc_start_main(main, argc, ubp_av, init, fini, rtld_fini, stack_end);
}
